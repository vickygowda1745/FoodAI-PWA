const CACHE_NAME = "foodai-cache-v10";

const STATIC_ASSETS = [
    "./",
    "./index.html",
    "./manifest.json"
];


// ============================================================
// INSTALL
// ============================================================

self.addEventListener(
    "install",
    event => {

        event.waitUntil(

            caches
                .open(CACHE_NAME)
                .then(cache => {

                    return cache.addAll(
                        STATIC_ASSETS
                    );

                })
                .catch(error => {

                    console.log(
                        "FoodAI cache install error:",
                        error
                    );

                })

        );

        // Activate this new service worker immediately.
        self.skipWaiting();

    }
);


// ============================================================
// ACTIVATE
// ============================================================

self.addEventListener(
    "activate",
    event => {

        event.waitUntil(

            caches
                .keys()
                .then(cacheNames => {

                    return Promise.all(

                        cacheNames.map(
                            cacheName => {

                                if (
                                    cacheName !== CACHE_NAME
                                ) {

                                    return caches.delete(
                                        cacheName
                                    );

                                }

                            }
                        )

                    );

                })
                .then(() => {

                    return self.clients.claim();

                })

        );

    }
);


// ============================================================
// FETCH
// ============================================================

self.addEventListener(
    "fetch",
    event => {

        const request =
            event.request;


        // Only handle GET requests.
        if (
            request.method !== "GET"
        ) {

            return;

        }


        const url =
            new URL(
                request.url
            );


        // Do not cache Render API calls.
        if (
            url.hostname.includes(
                "onrender.com"
            )
        ) {

            return;

        }


        // ====================================================
        // HTML / PAGE NAVIGATION
        // NETWORK FIRST
        // ====================================================

        if (
            request.mode === "navigate" ||
            request.destination === "document"
        ) {

            event.respondWith(

                fetch(
                    request,
                    {
                        cache:
                            "no-store"
                    }
                )
                    .then(response => {

                        const copy =
                            response.clone();


                        caches
                            .open(
                                CACHE_NAME
                            )
                            .then(cache => {

                                cache.put(
                                    request,
                                    copy
                                );

                            });


                        return response;

                    })
                    .catch(() => {

                        return caches.match(
                            "./index.html"
                        );

                    })

            );

            return;

        }


        // ====================================================
        // STATIC FILES
        // CACHE FIRST, THEN NETWORK
        // ====================================================

        event.respondWith(

            caches
                .match(
                    request
                )
                .then(cachedResponse => {

                    if (
                        cachedResponse
                    ) {

                        return cachedResponse;

                    }


                    return fetch(
                        request
                    )
                        .then(response => {

                            if (
                                !response ||
                                response.status !== 200 ||
                                response.type === "opaque"
                            ) {

                                return response;

                            }


                            const copy =
                                response.clone();


                            caches
                                .open(
                                    CACHE_NAME
                                )
                                .then(cache => {

                                    cache.put(
                                        request,
                                        copy
                                    );

                                });


                            return response;

                        });

                })

        );

    }
);