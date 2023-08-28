# onnx-deploy-gcp

Google Cloud Platform yordamida lokal holatda ishlatish uchun:

```
$ streamlit run gcp.py --server.port <port> &
$ ngrok http <port>
```

Serverda ishlatish uchun
```
$ tmux new-session -d -s <session_name1> "streamlit run gcp.py --server.port <port>"
$ tmux new-session -d -s <session_name2> "ngrok http <port>"
$ tmux attach -t <session_name>
```
