import http from "k6/http"
import { check, sleep } from "k6"

export const options = {
    stages: [
        { duration: "30s", target: 50 },
        { duration: "1m", target: 100 },
        { duration: "30s", target: 0 }
    ],
    thresholds: {
        http_req_failed: ["rate<0.01"],
        http_req_duration: ["p(95)<1000"],
    }
}

const API_URL = __ENV.API_URL || "http://localhost:8000/predict";
const API_KEY = __ENV.API_KEY || "dev-secret-key";

const payload = JSON.stringify({
    amt: 5000,
    lat: 40.7128,
    long: -74.006,
    merch_lat: 34.0522,
    merch_long: -118.2437,
    city_pop: 1000,
    category: "shopping_net",
    gender: "M",
    state: "CA",
    merchant: "unknown_rare_store",
    trans_date_trans_time: "2020-12-25 02:30:00",
    dob: "2003-01-01"
});

const params = {
    headers: {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }
}

export default function () {
    const response = http.post(API_URL, payload, params)

    check(response, {
        "status is 200": (r) => r.status === 200,
        "has fraud_probability": (r) => r.status === 200 && r.json("fraud_probability") !== undefined
    })

    sleep(0.1)
}