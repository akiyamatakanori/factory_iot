#!/bin/bash

# Factory AI Multi-Agent Communication System - HTTPS起動スクリプト

echo "🤖 Factory AI Multi-Agent Communication System を起動します..."
echo ""
echo "📋 システム情報:"
echo "  - URL: https://localhost:8501"
echo "  - プロトコル: HTTPS (SSL/TLS)"
echo "  - ポート: 8501"
echo "  - 特徴: AIエージェント間自律通信システム"
echo ""
echo "⚠️  注意: 初回アクセス時はブラウザで証明書の警告が表示されます"
echo "    「詳細設定」→「localhost にアクセスする(安全ではありません)」を選択してください"
echo ""
echo "🔄 6つのAIエージェントが自律的に通信を開始します..."
echo ""

# ブラウザを自動的に開く（3秒後）
echo "⏳ 3秒後にブラウザを自動的に開きます..."
(sleep 3 && open -a "Google Chrome" https://localhost:8501) &

# Streamlitアプリケーションを起動
streamlit run factory_ai_communication.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.sslCertFile=cert.pem \
    --server.sslKeyFile=key.pem \
    --browser.gatherUsageStats=false \
    --server.headless=true
