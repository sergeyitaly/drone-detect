// /public/sw.js
self.addEventListener('install', (event) => {
    event.waitUntil(self.skipWaiting());
  });
  
  self.addEventListener('activate', (event) => {
    event.waitUntil(self.clients.claim());
  });
  
  // Optional: Add fetch event handler for offline support
  self.addEventListener('fetch', (event) => {
    // You can add caching strategies here if needed
    event.respondWith(fetch(event.request));
  });