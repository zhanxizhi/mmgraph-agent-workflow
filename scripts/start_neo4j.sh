#!/bin/bash
# 启动 Neo4j 容器 (开发用)
# 用法: bash scripts/start_neo4j.sh [start|stop|restart|logs|status]

ACTION=${1:-start}

case "$ACTION" in
  start)
    # 如果容器已存在则直接启动，否则新建
    if docker ps -a --format '{{.Names}}' | grep -q '^mmgraph-neo4j$'; then
      echo "容器已存在，启动中..."
      docker start mmgraph-neo4j
    else
      echo "创建并启动 Neo4j 容器..."
      docker run -d \
        --name mmgraph-neo4j \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password \
        -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
        -e NEO4J_dbms_memory_heap_initial__size=512m \
        -e NEO4J_dbms_memory_heap_max__size=2G \
        -v mmgraph-neo4j-data:/data \
        neo4j:5.20
    fi
    echo ""
    echo "等待 Neo4j 就绪（约30秒）..."
    sleep 8
    until docker exec mmgraph-neo4j cypher-shell -u neo4j -p password "RETURN 1" > /dev/null 2>&1; do
      echo "  还在启动中..."
      sleep 5
    done
    echo "Neo4j 已就绪！"
    echo "  Web UI : http://localhost:7474"
    echo "  Bolt   : bolt://localhost:7687"
    echo "  账号   : neo4j / password"
    ;;
  stop)
    docker stop mmgraph-neo4j && echo "已停止"
    ;;
  restart)
    docker restart mmgraph-neo4j && echo "已重启"
    ;;
  logs)
    docker logs -f mmgraph-neo4j
    ;;
  status)
    docker inspect --format '{{.State.Status}}' mmgraph-neo4j 2>/dev/null || echo "容器不存在"
    ;;
  *)
    echo "用法: $0 [start|stop|restart|logs|status]"
    exit 1
    ;;
esac
