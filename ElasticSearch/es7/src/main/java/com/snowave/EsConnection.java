package com.snowave;

import java.io.IOException;
import java.util.function.Consumer;

import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.elasticsearch.client.RestHighLevelClient;

public class EsConnection {
    public static void connnect(Consumer<RestHighLevelClient> consumer) throws IOException {
        try (RestHighLevelClient client = new RestHighLevelClient(
                RestClient.builder(new HttpHost("localhost", 9200, "http")))) {
            consumer.accept(client);
        }
    }
}
