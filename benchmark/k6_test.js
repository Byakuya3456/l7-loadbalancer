import http from 'k6/http';
import { check } from 'k6';

export const options = {
    scenarios: {
        ai_routing_get: {
            executor: 'constant-vus',
            vus: 10,
            duration: '20s',
            exec: 'testGet',
        },
        ai_routing_post_varied: {
            executor: 'constant-vus',
            vus: 10,
            duration: '20s',
            exec: 'testPostVaried',
        },
        ai_waf_drop: {
            executor: 'constant-vus',
            vus: 10,
            duration: '20s',
            exec: 'testDelete',
        },
    },
};

export function testGet() {
    const res = http.get('http://127.0.0.1:8080/api/test');
    check(res, { 'status is 200': (r) => r.status === 200 });
}

export function testPostVaried() {
    // Generate random payload size between 100 bytes and 50 KB
    const isLarge = Math.random() > 0.5;
    const payloadSize = isLarge ? 40000 : 500; 
    
    // Create a random string of that length
    const randomString = 'A'.repeat(payloadSize);
    const payload = JSON.stringify({ data: randomString, size: isLarge ? 'large' : 'small' });
    
    const params = { headers: { 'Content-Type': 'application/json' } };
    const res = http.post('http://127.0.0.1:8080/api/upload', payload, params);
    
    // It should still return 200, but the AI will silently route it to a different backend internally
    check(res, { 'status is 200': (r) => r.status === 200 });
}

export function testDelete() {
    const res = http.del('http://127.0.0.1:8080/api/resource/42');
    check(res, { 'status is 403 (blocked)': (r) => r.status === 403 });
}
