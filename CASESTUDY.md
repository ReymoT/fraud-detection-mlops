# Building a Production-Style Fraud Inference Platform on Kubernetes

Early versions of the project explored unsupervised anomaly detection approaches including One-Class SVM, Isolation Forest, and Kernel Density Estimation.

The results of those models exposed an important lesson about imbalanced classification metrics.

Several models produced superficially strong accuracy and ROC-AUC values despite performing poorly on the actual fraud class. One-Class SVM achieved a ROC-AUC above 0.83 while still producing a PR-AUC of only 0.13. Isolation Forest performed even worse, with PR-AUC below 0.10 despite maintaining over 99% overall accuracy, and KDE achieved a 0.216 PR-AUC - the highest of the three.

The issue was class imbalance. With fraud rates at about 0.5%, ROC-AUC and accuracy can remain deceptively high even when minority-class precision and recall are operationally unusable. Precision/recall metrics exposed the real behavior far more clearly.

Kernel Density Estimation performed better than the other unsupervised approaches, but all anomaly detection variants struggled with unstable decision boundaries and high false positive sensitivity under shifting transaction distributions.

That experience influenced the architecture of this system significantly. Rather than pursuing increasingly complex anomaly detection pipelines, I shifted focus toward building a more operationally reliable supervised inference platform around XGBoost, with stronger observability, retraining workflows, and runtime safeguards. The project ultimately became less about squeezing out marginal model accuracy and more about understanding how ML systems behave under real serving constraints.

The original goal was to train a fraud classifier on over one million credit card transactions and expose it through a real-time inference API. The model itself was not especially novel, as gradient-boosted trees remain extremely strong on tabular fraud datasets, and XGBoost reached a PR-AUC of roughly 0.86 without any special architecture work.

The interesting problems emerged after the model and the basic version of the inference API worked.

Fraud detection creates a strange combination of constraints:

* the positive class is extremely rare (~0.5%)
* requests arrive in bursts rather than evenly (end-of-day settlement periods, promotional events, afternoon shopping, morning commutes, etc.)
* inference must happen in near real time
* latency matters operationally a lot
* explainability matters regulatorily
* model drift is inevitable because of changing spending habits, fraud patterns change

Training a classifier model solves only a small portion of the actual problem.

After completing the basic prototype of the model + inference API, I was determined to serve fraud predictions reliably under bursty, concurrent load while maintaining bounded latency and operational visibility.

# Real-Time Fraud Scoring Under Load

Fraud detection systems operate in a real-time environment. A transaction authorization request cannot wait for offline batch scoring. The inference path sits directly inside a payment flow, which means latency budgets are constrained by user experience and payment processor expectations.

Additionally, fraud traffic is inherently very bursty.

Large retail events, settlement periods, or even normal day-to-day afternoon shopping sprees can create sharp traffic spikes where inference throughput suddenly matters far more than average latency. A system optimized only for single-request latency may collapse in production under burst concurrency.

The goal of mine was to:

* minimize per-request latency
* maximize throughput under concurrency
* handle bursty traffic reliably
* prevent overload collapse
* preserve observability into runtime behavior


# Why XGBoost Instead of Deep Learning

The first architectural decision was model selection itself.

Neural networks are attractive because modern ML infrastructure discussions heavily emphasize deep learning systems. However, fraud detection on structured transactional data remains a domain where gradient-boosted trees are extremely competitive.

The dataset consisted primarily of:

* transaction amounts
* geolocation features
* merchant metadata
* temporal patterns
* demographic attributes

These are highly structured, heterogeneous tabular features. XGBoost handles this type of data exceptionally well with relatively small overhead.

A deep learning model choice would likely have:

* increased training complexity
* increased inference latency
* increased infrastructure requirements
* required GPU acceleration to justify itself
* more operational overhead in general

without outperforming gradient boosting by a meaningful margin on this dataset.

The simplicity of XGBoost mattered more than the buzz around deep learning.

This decision also simplified explainability. SHAP integrates naturally with tree ensembles and allowed prediction explanations to be returned per request. In financial crime/fraud systems, explainability is not an optional feature. Unjustified predictions become difficult to defend regulatorily or engineering-wise.

# Designing the Inference Runtime

The most important architectural decision by far in the project was the custom asynchronous batching engine.

The simplest implementation would have been a basic FastAPI endpoint with a synchronous `predict_proba` function.

That approach works under lighter load but scales very poorly because inference overhead becomes dominated by Python request handling, preprocessing and repeated model overhead rather than actual prediction computation.

Instead, the runtime was designed around dynamic batching.

Requests entering the API are placed into an asynchronous queue. A background batching worker groups requests arriving within a configurable timeout window and executes inference on the entire batch simultaneously.

So instead of executing a `predict_proba` function on every single incoming request individually, we handle bursts by calling the `predict_proba` function on an entire batch of requests.

Vectorized inference divides the overhead across many requests simultaneously.

The runtime implemented:

* bounded asynchronous queues
* configurable batch timeout windows
* configurable maximum batch size
* futures-based response resolution
* request timeout handling
* runtime metrics collection

The bounded queue was especially important. One of the easiest ways to destroy a real-time inference system is allowing unbounded queue growth under overload. Without limits, latency becomes nonlinear and eventually destabilizes the process itself through memory pressure and cascading timeouts.

When the queue fills:

* requests are rejected explicitly with HTTP 503
* latency remains bounded for accepted requests
* the system fails predictably rather than catastrophically

The tradeoff was that instead of silently degrading every request, shed the excess load predictably.

# Throughput vs Tail Latency

The batching runtime increased throughput from roughly 157 req/s to 187 req/s under 50 concurrent clients, yet the most interesting benchmark result was not the throughput increase itself, but the latency distribution shift introduced by batching.

The p95 latency improved, while the p99 latency worsened slightly.

At first this looked confusing and contradictory.

The explanation comes from queueing behavior.

Most requests benefited from batching because vectorized inference reduced per-request execution overhead substantially. CPU utilization became more efficient and majority of the requests completed faster.

However, by introducing a queue, we also introduced waiting time.

The worst-case requests are the ones that arrive immediately after a batch dispatches. Those requests must wait for the next timeout window, the next batch formation cycle and the next inference execution.

Most requests never experience this worst-case timing.

Those unlucky requests that do, however, do dominate p99 latency.

This is the inherent tradeoff when it comes to batching. Better throughput, better CPU efficiency, improved p95 latency for slightly worse tail latency

For fraud detection, this is absolutely acceptable because system stability under burst traffic mattered more than minimizing worst-case latency for every single request.

Additionally, an average batch size of approximately 43 requests suggested that the request arrival rate was sufficiently high for batching efficiency and that the queue was not catastrophically overloaded.

If average batch size had remained extremely small, the architectural decision of batching would simply just not justify itself. If batches constantly saturated maximum size, queue pressure would likely have become problematic.

# Kubernetes and Canary Rollouts

Once the runtime stabilized locally, the next problem became deployment behavior under orchestration.

The inference API was deployed to Google Kubernetes Engine using:

* stable/canary deployments
* Horizontal Pod Autoscaling
* Prometheus scraping
* Grafana observability
* readiness/liveness probes

The canary rollout implementation intentionally avoided service mesh complexity.

Instead of weighted routing, traffic splitting was performed through stable-canary replica ratio.

A 1:2 canary-to-stable ratio approximates a 33% canary traffic split without introducing more overhead, complexity and configuration burden.

The relatively high canary ratio was partly driven by GKE Autopilot quota constraints during testing. In a larger production environment, the canary would receive a way smaller percentage of traffic initially.

The tradeoff is precision.

Replica ratios provide only approximate routing percentages, while service meshes provide deterministic weighted routing. For this project I prioritized simplicity.

The pods also exposed a `release_version` field in API responses so rollout behavior could be validated directly during testing.

# What Broke

The most educational parts of the project were the failures.

One edge case I guarded against was timeout behavior in the async batching layer. A request can time out while still waiting inside the batching queue, while the background worker may complete inference later and attempt to resolve the same Future. This creates a race condition between request cancellation and batch completion. To make the runtime safe under load, the batcher checks `future.cancelled()` before calling `set_result()` or `set_exception()`.

Kubernetes autoscaling also behaved unexpectedly at first.

The Horizontal Pod Autoscaler repeatedly showed `cpu: <unknown>` despite metrics-server installation. The issue turned out to be metrics propagation timing combined with insufficient pod resource requests. HPA behavior in local Kubernetes environments proved significantly less deterministic than documentation suggests.

Google Kubernetes Engine introduced a different class of operational problems.

Deploying Grafana and Prometheus through the standard `kube-prometheus-stack` Helm chart failed repeatedly because GKE Autopilot restricts node-level monitoring components such as node-exporter and control plane scraping.

The eventual solution was abandoning the full stack in favor of lightweight standalone Prometheus and Grafana deployments focused only on application metrics.

Another failure came from overprovisioning.

Running multiple inference replicas, Prometheus, Grafana and the Streamlit dashboard simultaneously exceeded GKE quota limits and triggered unschedulable pods, failed scale-up events and CrashLoopBackOff behavior.

This became an important reminder that Kubernetes scheduling problems are often resource allocation problems rather the problems with application itself.

# Observability

One design decision that became increasingly important was exposing runtime metrics from the beginning.

The inference service exported throughput, queue depth, batch size, timeout counts, rejected requests and latency histograms through Prometheus-compatible endpoints.

Grafana dashboards made queue behavior visible during load testing.

Without observability, many runtime behaviors would have remained invisible such as queue saturation or latency spikes.

The monitoring stack changed debugging from guesswork into measurement.

# What I Would Do Differently

If rebuilding the platform from scratch, I would introduce infrastructure-as-code much earlier rather than retrofitting deployment automation later. Terraform would likely become the foundation for provisioning GKE, IAM, networking, and monitoring infrastructure.

I would also add distributed tracing from the beginning. Prometheus metrics were valuable for understanding aggregate system behavior, but tracing would provide significantly better visibility into latency propagation across Kafka ingestion, queueing, inference execution, and API response handling.

Finally, I would replace replica-ratio canary routing with a service mesh such as Istio. Replica-based traffic splitting was operationally simple, but weighted routing would allow far more precise rollout control and safer progressive deployments.

# What Comes Next

The next stage of the platform would focus on adaptive inference infrastructure rather than additional model experimentation.

A dedicated inference serving layer such as Triton Inference Server would allow GPU-backed dynamic batching, multi-model serving, and more advanced scheduling policies. I would also introduce online learning workflows driven by confirmed fraud feedback loops so the model could adapt more continuously to changing transaction behavior.

Operationally, a feature store and model approval workflow would help standardize feature consistency and deployment governance across retraining cycles. An explainability dashboard exposing SHAP distributions and drift metrics over time would also improve observability into model behavior beyond raw prediction performance.

The current architecture successfully demonstrates the operational realities of serving ML systems under load, but production-scale inference platforms become fundamentally scheduling and orchestration problems rather than modeling problems.