package com.snowave.elastic.search.app;

import java.io.IOException;

import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.ElasticsearchTransport;
import co.elastic.clients.transport.rest_client.RestClientTransport;

/**
 * 使用客户端的方式访问服务器
 */
public class App0 {
    /**
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        // 在es8中我们使用的就是ElasticsearchClient
        RestClient restClient = RestClient.builder(new HttpHost("localhost", 9200, "http")).build();
        ElasticsearchTransport transport = new RestClientTransport(restClient, new JacksonJsonpMapper());
        ElasticsearchClient client = new ElasticsearchClient(transport);

        // 关闭客户端
        transport.close();
        restClient.close();
    }
}
