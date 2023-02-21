package com.snowave.elastic.search.app;

import java.util.function.Consumer;
import java.util.function.Function;

import org.apache.http.HttpHost;
import org.apache.http.auth.AuthScope;
import org.apache.http.auth.UsernamePasswordCredentials;
import org.apache.http.client.CredentialsProvider;
import org.apache.http.impl.client.BasicCredentialsProvider;
import org.elasticsearch.client.RestClient;

import co.elastic.clients.elasticsearch.ElasticsearchAsyncClient;
import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.ElasticsearchTransport;
import co.elastic.clients.transport.rest_client.RestClientTransport;

public class ElasticSearchConnection {

    public static <T> T callWithResult(Function<ElasticsearchClient, T> job) throws Exception {
        // CredentialsProvider provider = new BasicCredentialsProvider();
        // provider.setCredentials(AuthScope.ANY, new
        // UsernamePasswordCredentials("elastic", "MagicWord"));
        try (RestClient restClient = RestClient
                .builder(new HttpHost("localhost", 9200, "http"))
                // .setHttpClientConfigCallback(builder ->
                // builder.setDefaultCredentialsProvider(provider)) // 添加密码
                .build();
                ElasticsearchTransport transport = new RestClientTransport(restClient, new JacksonJsonpMapper())) {
            return job.apply(new ElasticsearchClient(transport));
        }
    }

    public static void call(Consumer<ElasticsearchClient> job) throws Exception {
        // CredentialsProvider provider = new BasicCredentialsProvider();
        // provider.setCredentials(AuthScope.ANY, new
        // UsernamePasswordCredentials("elastic", "MagicWord"));
        try (RestClient restClient = RestClient
                .builder(new HttpHost("localhost", 9200, "http"))
                // .setHttpClientConfigCallback(builder ->
                // builder.setDefaultCredentialsProvider(provider)) // 添加密码
                .build();
                ElasticsearchTransport transport = new RestClientTransport(restClient, new JacksonJsonpMapper())) {
            job.accept(new ElasticsearchClient(transport));
        }
    }

    public static void callAsync(Consumer<ElasticsearchAsyncClient> job) throws Exception {
        // 用户密码登录器
        CredentialsProvider provider = new BasicCredentialsProvider();
        provider.setCredentials(AuthScope.ANY, new UsernamePasswordCredentials("elastic", "MagicWord"));

        // 添加证书
        // Path caCertPath = Paths.get("certs/es-ca-path.crt");
        // CertificateFactory factory = CertificateFactory.getInstance("X.509");
        // Certificate trustedCA;
        // try (InputStream is = Files.newInputStream(caCertPath)) {
        // trustedCA = factory.generateCertificates(is);
        // }
        // KeyStore trustStore = KeyStore.getInstance("pkcs12");
        // trustStore.load(null);
        // trustStore.setCertificateEntry("ca", trustedCA);
        // final SSLContextBuilder builder = SSLContext.loadTrustMaterial(trustStore,
        // null);
        // SSLContext sslContext = builder.build();

        try (RestClient restClient = RestClient
                .builder(new HttpHost("localhost", 9200, "http"))
                .setHttpClientConfigCallback(builder -> {
                    builder.setDefaultCredentialsProvider(provider);
                    // builder.setSSLHostnameVerifier(NoopHostnameVerifier.INSTANCE);
                    // builder.setSSLContext(sslContext);
                    return builder;
                })
                .build();
                ElasticsearchTransport transport = new RestClientTransport(restClient, new JacksonJsonpMapper())) {
            job.accept(new ElasticsearchAsyncClient(transport));
        }
    }

    public static ElasticsearchAsyncClient getAsyncClient() throws Exception {
        // 用户密码登录器
        CredentialsProvider provider = new BasicCredentialsProvider();
        provider.setCredentials(AuthScope.ANY, new UsernamePasswordCredentials("elastic", "MagicWord"));
        RestClient restClient = RestClient
                .builder(new HttpHost("localhost", 9200, "http"))
                .setHttpClientConfigCallback(builder -> builder.setDefaultCredentialsProvider(provider))
                .build();
        ElasticsearchTransport transport = new RestClientTransport(restClient, new JacksonJsonpMapper());
        return new ElasticsearchAsyncClient(transport);
    }
}
