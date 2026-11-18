BASE_DIR=$(cd "$(dirname "$0")"; pwd)
export NACOS_USERNAME="nacos"
export NACOS_PASSWORD="nacos"
export NACOS_ENABLED=true
export NACOS_SERVER_ADDRESSES="http://172.19.2.191:8848"
export NACOS_SERVICE_NAME="autoapex-timeseries-serving"

# 设置 PYTHONPATH 包含当前目录
export PYTHONPATH=$PYTHONPATH:$BASE_DIR

echo "Starting server from $BASE_DIR..."






export PORT=8000

# python "$BASE_DIR/main.py"

nohup python "$BASE_DIR/main.py" > main.log 2>&1 &