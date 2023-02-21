package com.snowave.elastic.search.app;

import co.elastic.clients.elasticsearch._types.query_dsl.MatchQuery;
import co.elastic.clients.elasticsearch._types.query_dsl.Query;
import co.elastic.clients.elasticsearch.core.SearchRequest;
import co.elastic.clients.elasticsearch.core.SearchResponse;

public class App_3_query {
    public static void main(String[] args) throws Exception {
        query2();
    }

    /**
     * 简单的match 查询
     * 
     * @throws Exception
     */
    public static void query1() throws Exception {
        String indexName = "myindex"; // 可以不指定索引
        ElasticSearchConnection.call((client) -> {
            try {
                MatchQuery matchQuery = new MatchQuery.Builder()
                        .field("age").query(30)
                        .build();
                Query query = new Query.Builder()
                        .match(matchQuery)
                        .build();

                SearchRequest searchRequest = new SearchRequest.Builder()
                        .query(query)
                        .build();
                SearchResponse searchResponse = client.search(searchRequest, Object.class);
                System.out.println("searchResponse: " + searchResponse);
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    /**
     * 使用lambda进行请求
     * 
     * @throws Exception
     */
    public static void query2() throws Exception {
        String indexName = "myindex"; // 可以不指定索引
        ElasticSearchConnection.call((client) -> {
            try {
                SearchResponse searchResponse = client.search(
                        req -> req.query(q -> q.match(m -> m.field("name").query("zhangsan"))),
                        Object.class);
                System.out.println("============================");
                System.out.println("searchResponse hits: " + searchResponse.hits());
                System.out.println("============================");
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

}
