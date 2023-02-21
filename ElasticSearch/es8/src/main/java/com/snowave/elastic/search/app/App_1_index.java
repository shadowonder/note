package com.snowave.elastic.search.app;

import java.io.IOException;

import co.elastic.clients.elasticsearch._types.ElasticsearchException;
import co.elastic.clients.elasticsearch.indices.CreateIndexRequest;
import co.elastic.clients.elasticsearch.indices.CreateIndexResponse;
import co.elastic.clients.elasticsearch.indices.DeleteIndexRequest;
import co.elastic.clients.elasticsearch.indices.DeleteIndexResponse;
import co.elastic.clients.elasticsearch.indices.ElasticsearchIndicesClient;
import co.elastic.clients.elasticsearch.indices.ExistsRequest;
import co.elastic.clients.elasticsearch.indices.GetIndexRequest;
import co.elastic.clients.elasticsearch.indices.GetIndexResponse;
import co.elastic.clients.elasticsearch.indices.IndexState;
import co.elastic.clients.transport.endpoints.BooleanResponse;

/**
 * 对index进行增删改查
 */
public class App_1_index {
    public static void main(String[] args) throws Exception {
        // indices();
        indicesLambda();
    }

    public static void indices() throws Exception {

        String indexName = "myindex";

        ElasticSearchConnection.call((client) -> {
            try {
                // 获取索引客户端对象
                ElasticsearchIndicesClient indices = client.indices();

                // 判断索引是否存在
                ExistsRequest existsRequest = new ExistsRequest.Builder().index(indexName).build();
                BooleanResponse existsResponse;
                existsResponse = indices.exists(existsRequest);
                boolean exists = existsResponse.value();
                if (exists) {
                    System.out.println("exists " + exists);
                } else {
                    // 需要采用构建器方式构建对象
                    CreateIndexRequest createIndexRequest = new CreateIndexRequest.Builder().index(indexName).build();
                    CreateIndexResponse createIndexResponse = indices.create(createIndexRequest);
                    // 创建索引的响应对象: CreateIndexResponse:
                    // {"index":"myindex","shards_acknowledged":true,"acknowledged":true}
                    System.out.println("创建索引的响应对象: " + createIndexResponse);
                    System.out.println(createIndexResponse.acknowledged());
                    System.out.println(createIndexResponse.index());
                }

                // 查询索引
                GetIndexRequest getIndexRequest = new GetIndexRequest.Builder().index(indexName).build();
                GetIndexResponse getIndexResponse = indices.get(getIndexRequest);
                // 响应结果: GetIndexResponse:
                // {"myindex":{"aliases":{},"mappings":{},"settings":{"index":{"number_of_shards":"1","number_of_replicas":"1","routing":{"allocation":{"include":{"_tier_preference":"data_content"}}},"provided_name":"myindex","creation_date":1676682762846,"uuid":"tDNhoQBvR1y0umy5Erlz6w","version":{"created":"7170999"}}}}}
                // System.out.println("响应结果: " + getIndexResponse);
                // 获取索引状态:
                IndexState indexState = getIndexResponse.get(indexName);
                System.out.println("索引状态: " + indexState);

                // 删除索引
                DeleteIndexRequest deleteIndexRequest = new DeleteIndexRequest.Builder().index(indexName).build();
                DeleteIndexResponse deleteIndexResponse = indices.delete(deleteIndexRequest);
                // 删除返回结果: DeleteIndexResponse: {"acknowledged":true}
                System.out.println("删除返回结果: " + deleteIndexResponse);

            } catch (ElasticsearchException | IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    public static void indicesLambda() throws Exception {
        ElasticSearchConnection.call((client) -> {
            try {
                boolean exists = client.indices().exists(req -> req.index("myindex")).value();
                if (exists) {
                    System.out.println("Lambda发现index存在");
                } else {
                    System.out.println("创建index");
                    CreateIndexResponse response = client.indices().create(req -> req.index("myindex"));
                    System.out.println("常见结果: " + response);
                }

                GetIndexResponse getIndexResponse = client.indices().get(req -> req.index("myindex"));
                System.out.println("Index信息: " + getIndexResponse);

                DeleteIndexResponse result = client.indices().delete(req -> req.index("myindex"));
                System.out.println("删除动作: " + result);
            } catch (ElasticsearchException | IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }
}
