package com.snowave.elastic.search.app;

import co.elastic.clients.elasticsearch.ElasticsearchAsyncClient;

/**
 * 一步操作的示例
 */
public class App_4_async_query {
    public static void main(String[] args) throws Exception {
        asyncTest();
    }

    public static void asyncTest() throws Exception {
        ElasticsearchAsyncClient asyncClient = ElasticSearchConnection.getAsyncClient();
        asyncClient.indices()
                .create(req -> req.index("newindex"))
                .thenApply(resp -> {
                    // thenApply 类似一个peek的方法, 在使用回调函数以前可以直接使用另一种方法
                    System.out.println("req: " + resp);
                    return resp;
                })
                .whenComplete((resp, err) -> {
                    if (resp != null) {
                        System.out.println("====================");
                        System.out.println(resp.acknowledged());
                        System.out.println("====================");
                    } else {
                        err.printStackTrace();
                    }
                });

        System.out.println("主线程继续运行....");
    }
}
