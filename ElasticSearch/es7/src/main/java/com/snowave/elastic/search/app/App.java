package com.snowave.elastic.search.app;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Set;

import javax.xml.transform.stream.StreamSource;

import org.elasticsearch.action.admin.cluster.health.ClusterHealthRequest;
import org.elasticsearch.action.admin.cluster.health.ClusterHealthResponse;
import org.elasticsearch.action.admin.indices.create.CreateIndexRequest;
import org.elasticsearch.action.admin.indices.create.CreateIndexResponse;
import org.elasticsearch.action.admin.indices.get.GetIndexRequest;
import org.elasticsearch.action.admin.indices.get.GetIndexResponse;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.common.io.stream.StreamInput;

import com.snowave.EsConnection;

/**
 *
 */
public class App {
    public static void main(String[] args) throws Exception {
        searchIndex();
    }

    // 创建一个名为user的索引
    public static void createIndex() throws IOException {
        EsConnection.connnect((client) -> {
            CreateIndexRequest request = new CreateIndexRequest("user");
            try {
                CreateIndexResponse response = client.indices().create(request, RequestOptions.DEFAULT);
                // 响应体中存在着多种属性, 这里的创建只返回我们需要的
                boolean acknowledged = response.isAcknowledged();
                System.out.println("索引操作: " + acknowledged);
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    /**
     * 查询索引
     * 
     * @throws IOException
     */
    public static void searchIndex() throws IOException {
        EsConnection.connnect((client) -> {
            try {
                client.indices().getIndex(new GetIndexRequest())
                        .actionGet()
                        .getIndices();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }
}
