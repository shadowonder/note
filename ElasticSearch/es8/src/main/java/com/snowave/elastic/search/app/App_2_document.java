package com.snowave.elastic.search.app;

import java.util.ArrayList;
import java.util.List;

import com.snowave.elastic.search.app.domain.User;

import co.elastic.clients.elasticsearch._types.Result;
import co.elastic.clients.elasticsearch.core.BulkRequest;
import co.elastic.clients.elasticsearch.core.BulkResponse;
import co.elastic.clients.elasticsearch.core.CreateRequest;
import co.elastic.clients.elasticsearch.core.CreateResponse;
import co.elastic.clients.elasticsearch.core.DeleteRequest;
import co.elastic.clients.elasticsearch.core.DeleteResponse;
import co.elastic.clients.elasticsearch.core.bulk.BulkOperation;
import co.elastic.clients.elasticsearch.core.bulk.CreateOperation;

/**
 * 对文档进行增删改查
 * 需要注意的是, 所有的增删改查依旧存在一个 CreateRequest.build()的构造器来进行, 也可以使用lambda表达式进行操作.
 * 按照个人喜好.
 */
public class App_2_document {
    public static void main(String[] args) throws Exception {
        bulkLambdaAdd();
    }

    public static void addDocument() throws Exception {
        String indexName = "myindex";

        User user = new User(1001, "zhangsan", 30);

        ElasticSearchConnection.call((client) -> {
            try {
                CreateRequest createRequest = new CreateRequest.Builder<User>()
                        .index(indexName)
                        .id("1001")
                        .document(user)
                        .build();
                CreateResponse createResponse = client.create(createRequest);
                // 创建文档的响应对象: CreateResponse:
                // {"_id":"1001","_index":"myindex","_primary_term":1,"result":"created","_seq_no":0,"_shards":{"failed":0.0,"successful":1.0,"total":2.0},"_version":1}
                System.out.println("创建文档的响应对象: " + createResponse);

            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    public static void addLambdaDocument() throws Exception {
        String indexName = "myindex";
        User user = new User(1001, "zhangsan", 30);
        ElasticSearchConnection.call((client) -> {
            try {
                Result result = client.create(req -> req
                        .index(indexName)
                        .id("1002")
                        .document(user)).result(); // result是createResponse中的一个方法, 返回一个reulst类型
                System.out.println(result);
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    /**
     * 批量添加用户数据
     * 
     * @throws Exception
     */
    public static void bulkAdd() throws Exception {
        String indexName = "myindex";
        ElasticSearchConnection.call((client) -> {
            try {
                List<BulkOperation> opts = new ArrayList<>();

                // 生成10个结果
                for (int i = 1; i <= 10; i++) {
                    CreateOperation<User> createOperation = new CreateOperation.Builder<User>()
                            .index(indexName)
                            .id("200" + i)
                            .document(new User(2000 + i, "zhangsan" + i, 30 + i))
                            .build();
                    BulkOperation operation = new BulkOperation.Builder().create(createOperation).build();
                    opts.add(operation);
                }

                BulkRequest bulkRequest = new BulkRequest.Builder()
                        .operations(opts)
                        .build();
                BulkResponse bulkResponse = client.bulk(bulkRequest);
                System.out.println("Bulk: " + bulkResponse);

            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    public static void bulkLambdaAdd() throws Exception {
        String indexName = "myindex";
        ElasticSearchConnection.call((client) -> {
            try {
                List<User> users = new ArrayList<>();

                // 生成10个结果
                for (int i = 1; i <= 10; i++) {
                    users.add(new User(3000 + i, "zhangsan" + i, 30 + i));
                }

                BulkResponse bulkResponse = client.bulk(req -> {
                    users.forEach(user -> {
                        req.operations(
                                bulkReq -> bulkReq.create(
                                        doc -> doc.index(indexName).id(user.getId().toString()).document(user)));
                    });
                    return req;
                });
                System.out.println("Bulk: " + bulkResponse);

            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    /**
     * 删除文档
     * 
     * @throws Exception
     */
    public static void deleteDocument() throws Exception {
        String indexName = "myindex";
        ElasticSearchConnection.call((client) -> {
            try {

                DeleteRequest deleteRequest = new DeleteRequest.Builder()
                        .index(indexName)
                        .id("2001")
                        .build();
                DeleteResponse dResponse = client.delete(deleteRequest);
                System.out.println("delete response: " + dResponse);

            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

    public static void deleteLambda() throws Exception {
        String indexName = "myindex";
        ElasticSearchConnection.call((client) -> {
            try {

                DeleteRequest deleteRequest = new DeleteRequest.Builder()
                        .index(indexName)
                        .id("2001")
                        .build();
                DeleteResponse dResponse = client.delete(deleteRequest);
                System.out.println("delete response: " + dResponse);

            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });
    }

}
