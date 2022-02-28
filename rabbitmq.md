# Rabbitmq

如果需要安装管理插件需要进入 rabbitmq-plugin enable rabbitmq_management 来安装管理平台插件. 管理端口 15672

默认消息监听端口为 5672

如果需要自己打包 dockerfile 可以添加 plugin

```dockerfile
FROM rabbitmq:3.7-management
RUN rabbitmq-plugins enable --offline rabbitmq_mqtt rabbitmq_federation_management rabbitmq_stomp
```

启动 docker-compose:

```yml
version: "2.2"
services:
  rabbitmq:
    image: rabbitmq:3-management
    container_name: rabbitmq
    hostname: rabbitmq
    volumes:
      - ./data:/var/lib/rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: mqusername
      RABBITMQ_DEFAULT_PASS: mqpassword
    ports:
      - 5672:5672
      - 15672:15672
```

## 使用 java 配置简单的生产者消费者

使用 java 配置:

```xml
<dependency>
    <groupId>com.rabbitmq</groupId>
    <artifactId>amqp-client</artifactId>
    <version>5.10.0</version>
</dependency>
```

创建一个简单的生产者

```java
public class Producer {

    private static final String QUEUE = "myQueue"; // 队列名称

    public static void main(String[] args) throws IOException {
        // 1. 建立新的连接
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("192.168.10.20");
        factory.setPort(5672);
        factory.setUsername("mqusername");
        factory.setPassword("mqpassword");
        // 设置虚拟机,每一个mq服务可以设置多个虚拟机,每一个虚拟机就相当于独立的mq
        factory.setVirtualHost("/");
        // 建立连接
        Connection connection = null;
        try {
            connection = factory.newConnection();
            // 创建会话通道, 通道将通向message exchange交换机
            Channel channel = connection.createChannel();
            // 配置一个队列(可以添加交换机)
            // String queue, boolean durable, boolean exclusive, boolean autoDelete,
            //                                 Map<String, Object> arguments
            /**
             * String queue, boolean durable, boolean exclusive, boolean autoDelete,
             *                                  Map<String, Object> arguments
             *  queue 队列名称
             *  durable 是否持久化 (放入硬盘, 重启后队列还在)
             *  exclusive 是否独占. 只允许在该链接中访问. 如果连接关闭, 队列删除. 如果设置为true, 可用为临时队列
             *  autoDelete 自动删除, 和exclusive都为true时就是临时队列
             *  arguments 扩展参数,比如存活时间等
             */
            channel.queueDeclare(QUEUE, true, false, false, null);

            // 发送消息, 使用默认交换机
            /**
             * @param exchange 指定交换机, 如果不指定就是用默认
             * @param routingKey 路由key,交换机会用路由key交换到指定队列, 默认交换机,routingkey需要为队列名称
             * @param props 消息属性
             * @param body 消息内容
             */
            String message = "hello this is my message";
            channel.basicPublish("", QUEUE, null, message.getBytes());
            System.out.println("Sent message: " + message);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            connection.close();
        }
    }
}
```

编写 consumer

```java
/**
 * 消费者
 */
public class Consumer {

    private static final String QUEUE = "myQueue"; // 队列名称

    public static void main(String[] args) throws IOException, TimeoutException {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("192.168.10.20");
        factory.setPort(5672);
        factory.setUsername("mqusername");
        factory.setPassword("mqpassword");
        factory.setVirtualHost("/");
        Connection connection = factory.newConnection();
        Channel channel = connection.createChannel();

        // 尝试声明队列, 因为如果生产者没有启动, 是没有队列可以监听的, 程序不报错
        channel.queueDeclare(QUEUE, true, false, false, null);

        // 监听队列
        /**
         * @param queue 队列
         * @param autoAck 自动返回, 当消息者收到消息后需要返回确认信息. 如果设置为true的话, 会自动确认. 否则就要编程回复.
         *                如果不确认,消息队列的消息就不会删除, 会一直接受
         * @param callback 消费方法, 当消费者收到消息, 需要执行的方法
         */
        channel.basicConsume(QUEUE, true, getConsumer(channel));

        // 注意, 如果不关闭, 就会一直监听
    }

    public static DefaultConsumer getConsumer(Channel channel) {
        // 实现消费方法
        DefaultConsumer consumer = new DefaultConsumer(channel) {
            /**
             * 当收到消息后方法调用
             * @param consumerTag 表示消息标签, 可以再监听队列设置,channel.basicConsume
             * @param envelope 信封, 通过信封可以获得交换机, 消息id(deliverTag, 用来表示消息, 大部分时候用来确认消息已接收)
             * @param properties 消息的属性, 在发送参数的时候可以设置basicprops
             * @param body 消息内容
             * @throws IOException
             */
            @Override
            public void handleDelivery(String consumerTag,
                                       Envelope envelope,
                                       AMQP.BasicProperties properties,
                                       byte[] body)
                    throws IOException {
                String message = new String(body);
                System.out.println("receive message: " + message);
            }
        };
        return consumer;
    }
}
```

---

## 工作模式

- work queue 工作模式, 一个消费者一个消费者队列. 消息不能被重复消费者. 使用轮训算法.
- pub/sub 模式. 发布订阅模式. 存在一个交换机. 消息进入交换机,然后交给多个队列, 多个消费者监听不同的队列. 一个消息就可以而被多个消费者接收到.

  ```java
  public class ProducerPubSub {

    private static final String QUEUE1 = "QUEUE1"; // 队列名称
    private static final String QUEUE2 = "QUEUE2"; // 队列名称
    private static final String EXCHANGE = "MyEXCHANGE"; // 交换机名称

    public static void main(String[] args) throws IOException {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("192.168.10.20");
        factory.setPort(5672);
        factory.setUsername("mqusername");
        factory.setPassword("mqpassword");
        factory.setVirtualHost("/");
        Connection connection = null;
        try {
            connection = factory.newConnection();
            Channel channel = connection.createChannel();
            // 声明两个队列
            channel.queueDeclare(QUEUE1, true, false, false, null);
            channel.queueDeclare(QUEUE2, true, false, false, null);

            // 声明交换机
            /**
             * exchange: 名称
             * type: 类型 在BuildinExchangeType:
             *     fanout: publish/sub
             *     direct: 路由工作模式
             *     topic: topic工作模式
             *     headers: 对用header工作模式
             */
            channel.exchangeDeclare(EXCHANGE, BuiltinExchangeType.FANOUT);

            // 交换机队列绑定
            /**
             * @param queue 队列名称
             * @param exchange 交换机名称
             * @param routingKey 路由key, 交换机根据路由的key的值将消息转发到队列,发布模式设置为空
             */
            channel.queueBind(QUEUE1, EXCHANGE, "");
            channel.queueBind(QUEUE2, EXCHANGE, "");

            String message = "hello inform message";

            channel.basicPublish(EXCHANGE, "", null, message.getBytes());
            System.out.println("Sent message: " + message);

            channel.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            connection.close();
        }
    }
  }
  ```

- Routing 模式 路由模式

  交换机会根据 routingkey 路由到不同的目标队列中

  ```java
  public class ProducerRouting {

    private static final String QUEUE1 = "QUEUE1"; // 队列名称
    private static final String QUEUE2 = "QUEUE2"; // 队列名称
    private static final String QUEUE3 = "QUEUE3"; // 队列名称
    private static final String ROUTING_KEY1 = "Key1"; // 设定routingkey
    private static final String ROUTING_KEY2 = "Key2"; // 设定routingkey
    private static final String EXCHANGE = "MyRoutingSwitch"; // 交换机名称

    public static void main(String[] args) throws IOException {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("192.168.10.20");
        factory.setPort(5672);
        factory.setUsername("mqusername");
        factory.setPassword("mqpassword");
        factory.setVirtualHost("/");
        Connection connection = null;
        try {
            connection = factory.newConnection();
            Channel channel = connection.createChannel();
            // 声明两个队列
            channel.queueDeclare(QUEUE1, true, false, false, null);
            channel.queueDeclare(QUEUE2, true, false, false, null);

            // 声明交换机, 这里配置了direct类型
            channel.exchangeDeclare(EXCHANGE, BuiltinExchangeType.DIRECT);

            // 交换机队列绑定
            channel.queueBind(QUEUE1, EXCHANGE, ROUTING_KEY1);
            channel.queueBind(QUEUE2, EXCHANGE, ROUTING_KEY2);
            channel.queueBind(QUEUE3, EXCHANGE, ROUTING_KEY2);

            // 根据routingkey 发送消息
            String message1 = "hello inform consumer set 1";
            channel.basicPublish(EXCHANGE, ROUTING_KEY1, null, message1.getBytes());

            String message2 = "hello inform consumer set 2";
            channel.basicPublish(EXCHANGE, ROUTING_KEY2, null, message2.getBytes());


            channel.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            connection.close();
        }
    }
  }
  ```

- topic 模式, 基于通配符交换. 相比于 routing 模式, 这个使用通配符模式匹配. 一种为#, 匹配一个词或者多个词. 另一种为\* 匹配一个字符.

  - 比如 inform.# 可以匹配 inform.sms, inform.ems, inform.ems.email
  - 比如 inform.\* 可以匹配 inform.sms, inform.ems 不能匹配最后一个,因为只能匹配一个词

  ```java
  public class ProducerTopic {

    private static final String QUEUE1 = "QUEUE1"; // 队列名称
    private static final String QUEUE2 = "QUEUE2"; // 队列名称
    private static final String ROUTING_KEY1 = "inform.#.email.#"; // 匹配 inform.email, inform.email.ems,inform.ems.email
    private static final String ROUTING_KEY2 = "inform.#.ems.#"; // inform.ems, inform.email.ems,inform.ems.email
    private static final String EXCHANGE = "MyRoutingSwitch"; // 交换机名称

    public static void main(String[] args) throws IOException {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("192.168.10.20");
        factory.setPort(5672);
        factory.setUsername("mqusername");
        factory.setPassword("mqpassword");
        factory.setVirtualHost("/");
        Connection connection = null;
        try {
            connection = factory.newConnection();
            Channel channel = connection.createChannel();
            // 声明两个队列
            channel.queueDeclare(QUEUE1, true, false, false, null);
            channel.queueDeclare(QUEUE2, true, false, false, null);

            // 声明交换机, 这里配置了direct类型
            channel.exchangeDeclare(EXCHANGE, BuiltinExchangeType.TOPIC);

            // 交换机队列绑定
            channel.queueBind(QUEUE1, EXCHANGE, ROUTING_KEY1);
            channel.queueBind(QUEUE2, EXCHANGE, ROUTING_KEY2);

            // 根据routingkey 发送消息
            String message1 = "hello inform consumer set 1";
            // 配置多类型routingkey
            channel.basicPublish(EXCHANGE, "inform.email", null, message1.getBytes());

            String message2 = "hello inform consumer set 2";
            channel.basicPublish(EXCHANGE, "inform.ems", null, message2.getBytes());

            channel.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            connection.close();
        }
    }
  }
  ```

- header 模式

  发送的时候不在发送 routingkey 属性,相对的,会发送一个 kv 属性, 也就是一个 map. 在 bind 中,指定一个 propsmap 属性 `channel.queueBind(QUEUE1, EXCHANGE, "", headerMap)`. 然后再消息发送的时候发送 header

- rcp 模式. 远程方法调用. 消息队列可以实现异步调用. 客户端生产消息. 然后发送给 rpc 队列, 执行目标方法,然后返回响应消息发送给另一个队列. 调用者就会监听另一个队列.

## springboot

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>
```

配置属性

```yml
spring:
  application:
    name: rabbitmq-spring
  rabbitmq:
    host: 192.168.10.20
    port: 5672
    username: mqusername
    password: mqpassword
    virtual-host: /
```

配置一个 config 类声明队列:

```java

@Configuration
public class RabbitmqConfig {
    public static String QUEUE_INFORM_EMAIL = "queue_inform_email";
    public static String EXCHANGE_TOPICS_INFORM = "exchange_topics_inform";
    public static String ROUTINGKEY_EMAIL = "inform.#.email.#";

    //声明队列
    @Bean
    public Queue queueEmail() {
        return new Queue(QUEUE_INFORM_EMAIL);
    }

    //声明Topic交换机
    @Bean
    Exchange topicExchange() {
        return ExchangeBuilder.topicExchange(EXCHANGE_TOPICS_INFORM).durable(true).build();
    }

    //将队列与Topic交换机进行绑定，并指定路由键
    @Bean
    Binding topicBindingEmail(@Qualifier("queueEmail") Queue queue, @Qualifier("topicExchange") Exchange exchange) {
        return BindingBuilder.bind(queue).to(exchange).with(ROUTINGKEY_EMAIL).noargs();
    }
}
```

生产者

```java
@Autowired
RabbitTemplate rabbitTemplate;

@Test
public void testSendEmail() {
    rabbitTemplate.convertAndSend(RabbitmqConfig.EXCHANGE_TOPICS_INFORM, "inform.email", "message to send");
}
```

消费者, 可以使用@RabbitListener(queue={<队列名名称>}) 来监听队列

```java
// 注意这个名字要final, 同时还可以获得message 和channel
@RabbitListener(queues = {RabbitmqConfig.EXCHANGE_TOPICS_INFORM})
public void testSendEmail(String messageFromRabbitmq, Message message, Channel channel) {
    System.out.println("message from string: " + messageFromRabbitmq);
    System.out.println("message : " + message.getBody());
    System.out.println("channel : " + channel.toString());
}
```
