import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Any
import uuid
import tiktoken  # トークンカウント用
import requests  # Ollama API用

# ページ設定
st.set_page_config(
    page_title="AI Multi-Agent Communication System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .communication-box {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    .agent-message {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .traffic-high {
        color: #dc3545;
        font-weight: bold;
    }
    .traffic-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .traffic-low {
        color: #28a745;
        font-weight: bold;
    }
    .message-flow {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background-color: #fff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# Ollama LLM統合
# ================================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

def call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Ollama APIを呼び出してLLM応答を取得"""
    try:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            return f"[Error: Ollama API returned status {response.status_code}]"
            
    except requests.exceptions.ConnectionError:
        return "[Error: Ollamaに接続できません。`ollama serve`が実行されているか確認してください]"
    except Exception as e:
        return f"[Error: {str(e)}]"

def check_ollama_status() -> Dict[str, Any]:
    """Ollamaの状態をチェック"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            llama_installed = any(OLLAMA_MODEL in model.get('name', '') for model in models)
            return {
                'status': 'running',
                'models': models,
                'llama_installed': llama_installed
            }
    except:
        pass
    
    return {
        'status': 'not_running',
        'models': [],
        'llama_installed': False
    }

# ================================
# トークン料金計算
# ================================
# 2024年11月時点の料金（USD per 1M tokens）
TOKEN_PRICES = {
    'gpt-4o': {'input': 2.50, 'output': 10.00},
    'claude-sonnet-4': {'input': 3.00, 'output': 15.00},
    'gemini-1.5-pro': {'input': 1.25, 'output': 5.00}
}

# 固定為替レート
USD_TO_JPY = 155

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """テキストのトークン数をカウント"""
    try:
        # GPT-4o/Claude/Geminiの近似としてcl100k_baseを使用
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(str(text))
        return len(tokens)
    except Exception as e:
        # フォールバック: 文字数 / 4 で近似
        return len(str(text)) // 4

def calculate_cost(tokens: int, model: str, message_type: str = 'input') -> float:
    """トークン数から料金を計算（USD）"""
    if model not in TOKEN_PRICES:
        return 0.0
    
    price_per_million = TOKEN_PRICES[model][message_type]
    return (tokens / 1_000_000) * price_per_million

# ================================
# AIエージェント基底クラス
# ================================
class AIAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.message_queue = []
        self.processing_log = []
        self.status = "idle"
        
    def send_message(self, to_agent: str, message_type: str, data: Dict, priority: str = "normal"):
        """他のエージェントにメッセージを送信"""
        # dataをJSON serializable形式に変換
        serializable_data = self._make_serializable(data)
        
        # JSON文字列化
        json_str = json.dumps(serializable_data)
        
        # トークン数を計算
        token_count = count_tokens(json_str)
        
        message = {
            "id": str(uuid.uuid4()),
            "from": self.name,
            "to": to_agent,
            "type": message_type,
            "data": serializable_data,
            "priority": priority,
            "timestamp": datetime.now(),
            "size_kb": len(json_str) / 1024,  # 通信量(KB)
            "tokens": token_count,  # トークン数
            "cost_gpt4o": calculate_cost(token_count, 'gpt-4o', 'input'),
            "cost_claude_sonnet": calculate_cost(token_count, 'claude-sonnet-4', 'input'),
            "cost_gemini_pro": calculate_cost(token_count, 'gemini-1.5-pro', 'input')
        }
        return message
    
    def _make_serializable(self, obj):
        """オブジェクトをJSON serializable形式に変換"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            # DataFrameを辞書形式に変換（列ごと）
            result = {}
            for col in obj.columns:
                if pd.api.types.is_datetime64_any_dtype(obj[col]):
                    result[col] = obj[col].astype(str).tolist()
                else:
                    result[col] = obj[col].tolist()
            return result
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        else:
            return obj
    
    def receive_message(self, message: Dict):
        """メッセージを受信"""
        self.message_queue.append(message)
        
    def process(self, input_data: Any) -> Dict:
        """データ処理（各エージェントで実装）"""
        raise NotImplementedError
        
    def log_action(self, action: str, details: str):
        """アクション記録"""
        self.processing_log.append({
            "timestamp": datetime.now(),
            "action": action,
            "details": details
        })

# ================================
# 各AIエージェントの実装
# ================================

class ProcessControlAgent(AIAgent):
    """プロセス制御エージェント"""
    def __init__(self):
        super().__init__("ProcessControl", "プロセス制御と最適化")
        self.target_temperature = 75.0
        self.target_pressure = 1013.0
        
    def process(self, sensor_data: pd.DataFrame) -> Dict:
        """プロセスデータを分析し、次のエージェントに指示（LLM統合）"""
        self.status = "processing"
        self.log_action("process_analysis", "センサーデータの分析開始")
        
        # 制御パラメータの計算（3つのラインの平均）
        avg_temp = (sensor_data['temperature_line1'].mean() + 
                   sensor_data['temperature_line2'].mean() + 
                   sensor_data['temperature_line3'].mean()) / 3
        avg_pressure = (sensor_data['pressure_pump1'].mean() + 
                       sensor_data['pressure_pump2'].mean() + 
                       sensor_data['pressure_pump3'].mean()) / 3
        temp_deviation = abs(avg_temp - self.target_temperature)
        pressure_deviation = abs(avg_pressure - self.target_pressure)
        
        # 🤖 LLMで制御パラメータを最適化
        llm_prompt = f"""あなたは工場のプロセス制御エキスパートです。

現在の状況:
- 平均温度: {avg_temp:.2f}°C（目標: {self.target_temperature}°C、偏差: {temp_deviation:.2f}°C）
- 平均圧力: {avg_pressure:.2f} hPa（目標: {self.target_pressure} hPa、偏差: {pressure_deviation:.2f} hPa）
- Line 1温度: {sensor_data['temperature_line1'].mean():.2f}°C
- Line 2温度: {sensor_data['temperature_line2'].mean():.2f}°C
- Line 3温度: {sensor_data['temperature_line3'].mean():.2f}°C

制御すべき点を3つ以内で簡潔に提案してください。"""

        llm_response = call_ollama(
            llm_prompt,
            system_prompt="あなたは工場プロセス制御の専門家です。簡潔に3つ以内で提案してください。"
        )
        
        self.log_action("llm_analysis", f"LLM分析結果: {llm_response[:100]}...")
        
        # データ収集エージェントへの指示を生成
        message_to_collector = self.send_message(
            to_agent="DataCollection",
            message_type="request_detailed_data",
            data={
                "reason": "制御パラメータの詳細分析が必要",
                "target_sensors": ["temperature_line1", "temperature_line2", "temperature_line3", 
                                 "pressure_pump1", "pressure_pump2", "pressure_pump3", 
                                 "vibration_motor1", "vibration_motor2", "vibration_motor3"],
                "sampling_rate": "high" if temp_deviation > 5 else "normal",
                "time_window": "last_10_minutes",
                "llm_recommendation": llm_response
            },
            priority="high" if temp_deviation > 10 else "normal"
        )
        
        # 異常検知エージェントへの通知
        message_to_anomaly = self.send_message(
            to_agent="AnomalyDetection",
            message_type="control_status_update",
            data={
                "temperature_status": "warning" if temp_deviation > 5 else "normal",
                "pressure_status": "warning" if pressure_deviation > 10 else "normal",
                "control_actions_taken": ["temperature_adjustment"] if temp_deviation > 5 else [],
                "llm_analysis": llm_response
            },
            priority="medium"
        )
        
        self.status = "completed"
        return {
            "messages_sent": [message_to_collector, message_to_anomaly],
            "analysis": {
                "avg_temp": avg_temp,
                "avg_pressure": avg_pressure,
                "temp_deviation": temp_deviation,
                "pressure_deviation": pressure_deviation
            }
        }

class DataCollectionAgent(AIAgent):
    """データ収集エージェント"""
    def __init__(self):
        super().__init__("DataCollection", "センサーデータの収集と前処理")
        self.collection_rate = 10000  # pts/sec（10倍に増加）
        
    def process(self, request_message: Dict = None) -> Dict:
        """データ収集とエージェント間通信（大規模データ対応）"""
        self.status = "processing"
        self.log_action("data_collection", "大規模センサーデータ収集開始")
        
        # 💥 データ量を大幅に増加：1,000データポイント × 20センサー
        n_points = 1000  # 100 → 1000に増加
        n_sensors = 20    # 6 → 20センサーに増加
        
        timestamps = pd.date_range(end=datetime.now(), periods=n_points, freq='10s')
        
        # 基本センサー
        sensor_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature_line1': 75 + np.random.randn(n_points) * 3,
            'temperature_line2': 73 + np.random.randn(n_points) * 2.5,
            'temperature_line3': 76 + np.random.randn(n_points) * 3.2,
            'pressure_pump1': 1013 + np.random.randn(n_points) * 5,
            'pressure_pump2': 1010 + np.random.randn(n_points) * 4.8,
            'pressure_pump3': 1015 + np.random.randn(n_points) * 5.2,
            'vibration_motor1': 0.5 + np.abs(np.random.randn(n_points) * 0.1),
            'vibration_motor2': 0.48 + np.abs(np.random.randn(n_points) * 0.09),
            'vibration_motor3': 0.52 + np.abs(np.random.randn(n_points) * 0.11),
            'power_line1': 250 + np.random.randn(n_points) * 20,
            'power_line2': 245 + np.random.randn(n_points) * 18,
            'power_line3': 255 + np.random.randn(n_points) * 22,
            'production_rate_line1': 95 + np.random.randn(n_points) * 3,
            'production_rate_line2': 93 + np.random.randn(n_points) * 2.8,
            'production_rate_line3': 97 + np.random.randn(n_points) * 3.2,
            'humidity': 45 + np.random.randn(n_points) * 5,
            'air_quality': 80 + np.random.randn(n_points) * 10,
            'noise_level': 65 + np.random.randn(n_points) * 8,
            'flow_rate': 100 + np.random.randn(n_points) * 15,
            'rotation_speed': 1500 + np.random.randn(n_points) * 50
        })
        
        # 異常値を追加（5%）
        anomaly_indices = np.random.choice(n_points, size=int(n_points * 0.05), replace=False)
        for idx in anomaly_indices:
            sensor_data.loc[idx, 'temperature_line1'] += np.random.choice([-15, 15])
            sensor_data.loc[idx, 'vibration_motor1'] += 0.5
        
        # 💾 センサーデータをCSVファイルに保存
        try:
            save_dir = "/Users/takaakiy/factory_iot/SensorData"
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{save_dir}/sensor_data_{timestamp_str}.csv"
            sensor_data.to_csv(csv_filename, index=False)
            self.log_action("data_saved", f"センサーデータを保存: {csv_filename}")
        except Exception as e:
            self.log_action("data_save_error", f"保存エラー: {str(e)}")
        
        # プロセス制御エージェントへのレスポンス
        message_to_process = self.send_message(
            to_agent="ProcessControl",
            message_type="data_ready",
            data={
                "data_summary": {
                    "points_collected": len(sensor_data),
                    "sensors": list(sensor_data.columns),
                    "quality": "high",
                    "collection_rate": self.collection_rate,
                    "total_datapoints": len(sensor_data) * len(sensor_data.columns),
                    "saved_to": csv_filename if 'csv_filename' in locals() else "N/A"
                }
            },
            priority="normal"
        )
        
        # 💥 異常検知エージェントへ大容量データ転送
        message_to_anomaly = self.send_message(
            to_agent="AnomalyDetection",
            message_type="sensor_data_batch",
            data={
                "raw_data": {
                    col: sensor_data[col].tolist() for col in sensor_data.columns
                },
                "metadata": {
                    "collection_start": timestamps[0].isoformat(),
                    "collection_end": timestamps[-1].isoformat(),
                    "total_points": len(sensor_data),
                    "total_sensors": len(sensor_data.columns),
                    "sampling_interval": "10s",
                    "data_quality_score": 0.98
                }
            },
            priority="high"
        )
        
        # 品質分析エージェントへもデータ送信（3つの生産ライン）
        message_to_quality = self.send_message(
            to_agent="QualityAnalysis",
            message_type="production_data",
            data={
                "production_rate_line1": sensor_data['production_rate_line1'].tolist(),
                "production_rate_line2": sensor_data['production_rate_line2'].tolist(),
                "production_rate_line3": sensor_data['production_rate_line3'].tolist(),
                "quality_indicators": {
                    "temperature_variance_line1": float(sensor_data['temperature_line1'].var()),
                    "temperature_variance_line2": float(sensor_data['temperature_line2'].var()),
                    "temperature_variance_line3": float(sensor_data['temperature_line3'].var()),
                    "pressure_stability_pump1": float(sensor_data['pressure_pump1'].std()),
                    "pressure_stability_pump2": float(sensor_data['pressure_pump2'].std()),
                    "pressure_stability_pump3": float(sensor_data['pressure_pump3'].std()),
                    "humidity_average": float(sensor_data['humidity'].mean()),
                    "air_quality_average": float(sensor_data['air_quality'].mean())
                }
            },
            priority="normal"
        )
        
        self.status = "completed"
        return {
            "messages_sent": [message_to_process, message_to_anomaly, message_to_quality],
            "data": sensor_data
        }

class AnomalyDetectionAgent(AIAgent):
    """異常検知エージェント"""
    def __init__(self):
        super().__init__("AnomalyDetection", "異常パターンの検出と分析")
        
    def process(self, sensor_data: pd.DataFrame) -> Dict:
        """異常検知とアラート生成"""
        self.status = "processing"
        self.log_action("anomaly_detection", "異常検知アルゴリズム実行")
        
        # 異常検知（複数ラインの統合）
        # 温度異常（3つのラインを統合）
        temp_line1_threshold = (sensor_data['temperature_line1'].mean() - 2*sensor_data['temperature_line1'].std(),
                               sensor_data['temperature_line1'].mean() + 2*sensor_data['temperature_line1'].std())
        temp_anomalies_line1 = sensor_data[(sensor_data['temperature_line1'] < temp_line1_threshold[0]) | 
                                           (sensor_data['temperature_line1'] > temp_line1_threshold[1])]
        
        temp_line2_threshold = (sensor_data['temperature_line2'].mean() - 2*sensor_data['temperature_line2'].std(),
                               sensor_data['temperature_line2'].mean() + 2*sensor_data['temperature_line2'].std())
        temp_anomalies_line2 = sensor_data[(sensor_data['temperature_line2'] < temp_line2_threshold[0]) | 
                                           (sensor_data['temperature_line2'] > temp_line2_threshold[1])]
        
        temp_line3_threshold = (sensor_data['temperature_line3'].mean() - 2*sensor_data['temperature_line3'].std(),
                               sensor_data['temperature_line3'].mean() + 2*sensor_data['temperature_line3'].std())
        temp_anomalies_line3 = sensor_data[(sensor_data['temperature_line3'] < temp_line3_threshold[0]) | 
                                           (sensor_data['temperature_line3'] > temp_line3_threshold[1])]
        
        # 全温度異常を統合
        temp_anomalies = pd.concat([temp_anomalies_line1, temp_anomalies_line2, temp_anomalies_line3]).drop_duplicates()
        
        # 振動異常（3つのモーターを統合）
        vib_motor1_threshold = sensor_data['vibration_motor1'].mean() + 2*sensor_data['vibration_motor1'].std()
        vib_anomalies_motor1 = sensor_data[sensor_data['vibration_motor1'] > vib_motor1_threshold]
        
        vib_motor2_threshold = sensor_data['vibration_motor2'].mean() + 2*sensor_data['vibration_motor2'].std()
        vib_anomalies_motor2 = sensor_data[sensor_data['vibration_motor2'] > vib_motor2_threshold]
        
        vib_motor3_threshold = sensor_data['vibration_motor3'].mean() + 2*sensor_data['vibration_motor3'].std()
        vib_anomalies_motor3 = sensor_data[sensor_data['vibration_motor3'] > vib_motor3_threshold]
        
        # 全振動異常を統合
        vib_anomalies = pd.concat([vib_anomalies_motor1, vib_anomalies_motor2, vib_anomalies_motor3]).drop_duplicates()
        
        anomalies_detected = len(temp_anomalies) + len(vib_anomalies)
        
        messages = []
        
        # 🤖 LLMで異常原因を分析（常に実行）
        llm_prompt = f"""あなたは工場の異常検知エキスパートです。

検出された異常:
- 温度異常: {len(temp_anomalies)}件
- 振動異常: {len(vib_anomalies)}件
- 総異常数: {anomalies_detected}件

センサー統計:
- Line 1平均温度: {sensor_data['temperature_line1'].mean():.2f}°C
- Line 2平均温度: {sensor_data['temperature_line2'].mean():.2f}°C
- Line 3平均温度: {sensor_data['temperature_line3'].mean():.2f}°C
- Motor 1平均振動: {sensor_data['vibration_motor1'].mean():.3f}

{"異常が検出されました。" if anomalies_detected > 0 else "正常範囲内です。"}考えられる原因を2-3個挙げて、それぞれの対策を簡潔に提案してください。"""

        llm_analysis = call_ollama(
            llm_prompt,
            system_prompt="あなたは工場異常検知の専門家です。原因と対策を簡潔に2-3個提案してください。"
        )
        
        self.log_action("llm_anomaly_analysis", f"LLM異常分析: {llm_analysis[:100]}...")
        
        # 異常が検出された場合
        if anomalies_detected > 0:
            # 警告・通知エージェントへ即座にアラート送信（高優先度）
            message_to_alert = self.send_message(
                to_agent="AlertNotification",
                message_type="anomaly_alert",
                data={
                    "severity": "high" if anomalies_detected > 5 else "medium",
                    "anomalies": {
                        "temperature_anomalies": len(temp_anomalies),
                        "vibration_anomalies": len(vib_anomalies)
                    },
                    "recommended_actions": ["immediate_inspection", "process_adjustment"],
                    "affected_equipment": ["line_2", "pump_3"],
                    "llm_root_cause_analysis": llm_analysis
                },
                priority="urgent"
            )
            messages.append(message_to_alert)
            
            # 品質分析エージェントへ異常情報を送信
            message_to_quality = self.send_message(
                to_agent="QualityAnalysis",
                message_type="anomaly_impact_data",
                data={
                    "anomaly_periods": temp_anomalies['timestamp'].astype(str).tolist() if len(temp_anomalies) > 0 else [],
                    "impact_severity": "high" if anomalies_detected > 5 else "medium",
                    "potential_quality_impact": True,
                    "llm_analysis": llm_analysis
                },
                priority="high"
            )
            messages.append(message_to_quality)
            
            # 予測保全エージェントへリスク情報を送信
            message_to_maintenance = self.send_message(
                to_agent="PredictiveMaintenance",
                message_type="risk_indicator_update",
                data={
                    "vibration_trend": "increasing",
                    "temperature_instability": True,
                    "failure_risk_increase": 0.15
                },
                priority="high"
            )
            messages.append(message_to_maintenance)
        
        # プロセス制御エージェントへフィードバック
        message_to_process = self.send_message(
            to_agent="ProcessControl",
            message_type="anomaly_feedback",
            data={
                "anomaly_count": anomalies_detected,
                "control_effectiveness": "good" if anomalies_detected < 3 else "needs_improvement",
                "suggestions": ["tighten_temperature_control"] if len(temp_anomalies) > 2 else []
            },
            priority="normal"
        )
        messages.append(message_to_process)
        
        self.status = "completed"
        return {
            "messages_sent": messages,
            "anomalies": {
                "temp_anomalies": temp_anomalies,
                "vib_anomalies": vib_anomalies,
                "total_count": anomalies_detected
            }
        }

class QualityAnalysisAgent(AIAgent):
    """品質分析エージェント"""
    def __init__(self):
        super().__init__("QualityAnalysis", "製品品質の分析と改善提案")
        
    def process(self, production_data: Dict, anomaly_data: Dict = None) -> Dict:
        """品質分析とAI改善提案（LLM統合）"""
        self.status = "processing"
        self.log_action("quality_analysis", "品質指標の計算開始")
        
        # 品質指標計算
        quality_score = np.random.uniform(92, 98)
        defect_rate = np.random.uniform(0.5, 2.5)
        
        # 異常の影響を考慮
        if anomaly_data and anomaly_data.get('potential_quality_impact'):
            quality_score -= 2
            defect_rate += 0.5
        
        # 🤖 LLMで品質改善策を提案
        llm_prompt = f"""あなたは製造品質管理のエキスパートです。

現在の品質状況:
- 品質スコア: {quality_score:.2f}点（目標: 95点以上）
- 不良率: {defect_rate:.2f}%
- 生産速度: {production_data.get('rate', 95):.2f} units/min
- 異常検出: {"あり" if anomaly_data and anomaly_data.get('potential_quality_impact') else "なし"}

{"品質が目標を下回っています。" if quality_score < 95 else "品質は良好です。"}具体的な改善策を3つ提案してください。各提案は簡潔に1-2文で。"""

        llm_improvement = call_ollama(
            llm_prompt,
            system_prompt="あなたは製造品質の専門家です。実行可能な改善策を3つ、簡潔に提案してください。"
        )
        
        self.log_action("llm_quality_improvement", f"LLM改善提案: {llm_improvement[:100]}...")
        
        messages = []
        
        # 品質スコアが低い場合、複数エージェントへ通知
        if quality_score < 95:
            # 警告・通知エージェントへアラート
            message_to_alert = self.send_message(
                to_agent="AlertNotification",
                message_type="quality_alert",
                data={
                    "severity": "medium",
                    "quality_score": quality_score,
                    "defect_rate": defect_rate,
                    "trend": "declining",
                    "llm_improvement_plan": llm_improvement
                },
                priority="high"
            )
            messages.append(message_to_alert)
            
            # プロセス制御エージェントへ改善要求
            message_to_process = self.send_message(
                to_agent="ProcessControl",
                message_type="quality_improvement_request",
                data={
                    "target_parameters": ["temperature_stability", "pressure_consistency"],
                    "required_improvement": 3.0,
                    "recommendations": [
                        "reduce_temperature_variance",
                        "optimize_heating_cycle"
                    ],
                    "llm_analysis": llm_improvement
                },
                priority="high"
            )
            messages.append(message_to_process)
        
        # データ収集エージェントへ追加データ要求
        message_to_data = self.send_message(
            to_agent="DataCollection",
            message_type="request_quality_data",
            data={
                "data_points": ["product_dimensions", "surface_quality", "material_properties"],
                "sampling_frequency": "every_10_units"
            },
            priority="normal"
        )
        messages.append(message_to_data)
        
        # 予測保全エージェントへ品質トレンド情報を送信
        message_to_maintenance = self.send_message(
            to_agent="PredictiveMaintenance",
            message_type="quality_trend_data",
            data={
                "quality_degradation": quality_score < 95,
                "correlation_with_equipment": True,
                "maintenance_impact_prediction": "high"
            },
            priority="medium"
        )
        messages.append(message_to_maintenance)
        
        self.status = "completed"
        return {
            "messages_sent": messages,
            "analysis": {
                "quality_score": quality_score,
                "defect_rate": defect_rate
            }
        }

class AlertNotificationAgent(AIAgent):
    """警告・通知エージェント"""
    def __init__(self):
        super().__init__("AlertNotification", "アラート管理と通知配信")
        self.active_alerts = []
        
    def process(self, alert_data: Dict) -> Dict:
        """アラート処理と通知配信（LLM統合）"""
        self.status = "processing"
        self.log_action("alert_processing", f"アラート処理: {alert_data.get('severity', 'unknown')}")
        
        alert_id = str(uuid.uuid4())
        self.active_alerts.append({
            "id": alert_id,
            "data": alert_data,
            "timestamp": datetime.now()
        })
        
        # 🤖 LLMで通知メッセージを生成
        anomalies_info = alert_data.get('anomalies', {})
        quality_info = alert_data.get('quality_score', 'N/A')
        
        llm_prompt = f"""あなたは工場管理者への通知を作成する専門家です。

アラート情報:
- 重要度: {alert_data.get('severity', 'medium')}
- 異常検出数: {anomalies_info if isinstance(anomalies_info, dict) else 'N/A'}
- 品質スコア: {quality_info}
- 影響範囲: {alert_data.get('affected_equipment', [])}

管理者向けに、状況を簡潔に説明し、推奨アクションを3つ箇条書きで提案してください。全体で5文以内。"""

        llm_notification = call_ollama(
            llm_prompt,
            system_prompt="あなたは工場管理者への通知作成の専門家です。簡潔明瞭に、5文以内で状況と対策を伝えてください。"
        )
        
        self.log_action("llm_notification", f"LLM通知文: {llm_notification[:100]}...")
        
        messages = []
        
        # 重要度に応じて複数の通知チャネルへ配信
        if alert_data.get('severity') == 'high' or alert_data.get('severity') == 'urgent':
            # プロセス制御エージェントへ緊急停止要求
            message_to_process = self.send_message(
                to_agent="ProcessControl",
                message_type="emergency_action_required",
                data={
                    "alert_id": alert_id,
                    "action": "reduce_production_rate",
                    "reason": alert_data.get('anomalies', {}),
                    "llm_notification": llm_notification
                },
                priority="urgent"
            )
            messages.append(message_to_process)
            
            # 予測保全エージェントへ緊急メンテナンス要求
            message_to_maintenance = self.send_message(
                to_agent="PredictiveMaintenance",
                message_type="urgent_maintenance_request",
                data={
                    "alert_id": alert_id,
                    "equipment": alert_data.get('affected_equipment', []),
                    "urgency": "immediate",
                    "llm_notification": llm_notification
                },
                priority="urgent"
            )
            messages.append(message_to_maintenance)
        
        # データ収集エージェントへ監視強化要求
        message_to_data = self.send_message(
            to_agent="DataCollection",
            message_type="increase_monitoring",
            data={
                "target_sensors": ["temperature", "vibration", "pressure"],
                "duration": "30_minutes",
                "sampling_rate": "maximum",
                "alert_context": llm_notification
            },
            priority="high"
        )
        messages.append(message_to_data)
        
        self.status = "completed"
        return {
            "messages_sent": messages,
            "alert_id": alert_id,
            "notifications_sent": ["email", "slack", "sms"] if alert_data.get('severity') == 'high' else ["email"]
        }

class PredictiveMaintenanceAgent(AIAgent):
    """予測保全エージェント"""
    def __init__(self):
        super().__init__("PredictiveMaintenance", "故障予測と保全計画")
        self.equipment_status = {}
        
    def process(self, risk_data: Dict) -> Dict:
        """故障予測と保全計画の生成（LLM統合）"""
        self.status = "processing"
        self.log_action("predictive_analysis", "故障リスク分析開始")
        
        # リスクスコア計算
        base_risk = np.random.uniform(20, 40)
        if risk_data.get('vibration_trend') == 'increasing':
            base_risk += 15
        if risk_data.get('temperature_instability'):
            base_risk += 10
        
        risk_score = min(100, base_risk)
        days_to_maintenance = max(7, int(30 - risk_score / 3))
        
        # 🤖 LLMで保全計画を生成
        llm_prompt = f"""あなたは設備保全の専門家です。

予測結果:
- リスクスコア: {risk_score:.1f}%
- 推奨保全まで: {days_to_maintenance}日
- 振動トレンド: {risk_data.get('vibration_trend', 'stable')}
- 温度不安定: {"あり" if risk_data.get('temperature_instability') else "なし"}
- 対象設備: pump_3, conveyor_1

{"リスクが高いです。" if risk_score > 60 else "リスクは中程度です。"}具体的な保全計画を3ステップで提案してください。各ステップは簡潔に1文で。"""

        llm_maintenance_plan = call_ollama(
            llm_prompt,
            system_prompt="あなたは設備保全の専門家です。実行可能な保全計画を3ステップ、簡潔に提案してください。"
        )
        
        self.log_action("llm_maintenance_plan", f"LLM保全計画: {llm_maintenance_plan[:100]}...")
        
        messages = []
        
        # リスクが高い場合、複数エージェントへ警告
        if risk_score > 60:
            # 警告・通知エージェントへ保全アラート
            message_to_alert = self.send_message(
                to_agent="AlertNotification",
                message_type="maintenance_alert",
                data={
                    "severity": "high" if risk_score > 80 else "medium",
                    "risk_score": risk_score,
                    "days_to_maintenance": days_to_maintenance,
                    "equipment": ["pump_3", "conveyor_1"],
                    "llm_maintenance_plan": llm_maintenance_plan
                },
                priority="high"
            )
            messages.append(message_to_alert)
            
            # プロセス制御エージェントへ負荷軽減要求
            message_to_process = self.send_message(
                to_agent="ProcessControl",
                message_type="reduce_equipment_load",
                data={
                    "reason": "high_failure_risk",
                    "target_equipment": ["pump_3"],
                    "recommended_load_reduction": 0.2,
                    "llm_plan": llm_maintenance_plan
                },
                priority="high"
            )
            messages.append(message_to_process)
        
        # 品質分析エージェントへ保全影響を通知
        message_to_quality = self.send_message(
            to_agent="QualityAnalysis",
            message_type="maintenance_schedule_update",
            data={
                "scheduled_maintenance": {
                    "date": (datetime.now() + timedelta(days=days_to_maintenance)).isoformat(),
                    "duration": "4_hours",
                    "affected_lines": ["line_2"]
                },
                "production_impact": "medium",
                "llm_maintenance_plan": llm_maintenance_plan
            },
            priority="normal"
        )
        messages.append(message_to_quality)
        
        # データ収集エージェントへ予測モデル用データ要求
        message_to_data = self.send_message(
            to_agent="DataCollection",
            message_type="request_historical_data",
            data={
                "time_range": "last_30_days",
                "sensors": ["vibration", "temperature", "operating_hours"],
                "purpose": "model_training",
                "llm_context": llm_maintenance_plan
            },
            priority="low"
        )
        messages.append(message_to_data)
        
        self.status = "completed"
        return {
            "messages_sent": messages,
            "prediction": {
                "risk_score": risk_score,
                "days_to_maintenance": days_to_maintenance
            }
        }

# ================================
# セッション状態の初期化
# ================================
if 'agents' not in st.session_state:
    st.session_state.agents = {
        'ProcessControl': ProcessControlAgent(),
        'DataCollection': DataCollectionAgent(),
        'AnomalyDetection': AnomalyDetectionAgent(),
        'QualityAnalysis': QualityAnalysisAgent(),
        'AlertNotification': AlertNotificationAgent(),
        'PredictiveMaintenance': PredictiveMaintenanceAgent()
    }

if 'communication_log' not in st.session_state:
    st.session_state.communication_log = []

if 'total_traffic_kb' not in st.session_state:
    st.session_state.total_traffic_kb = 0

# 承認フロー用のセッション状態
if 'workflow_step' not in st.session_state:
    st.session_state.workflow_step = 0

if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = {}

if 'waiting_for_approval' not in st.session_state:
    st.session_state.waiting_for_approval = False

# ================================
# メイン処理フロー（ステップバイステップ承認）
# ================================
def execute_workflow_step(step: int):
    """指定されたステップを実行"""
    
    if step == 0:
        # ステップ0: 初期化
        st.session_state.workflow_data = {}
        st.session_state.workflow_step = 1
        return "ワークフロー開始準備完了"
    
    elif step == 1:
        # ステップ1: データ収集エージェント
        data_agent = st.session_state.agents['DataCollection']
        data_result = data_agent.process()
        sensor_data = data_result['data']
        
        # 通信ログに記録
        for msg in data_result['messages_sent']:
            st.session_state.communication_log.append(msg)
            st.session_state.total_traffic_kb += msg['size_kb']
        
        # 次のステップのためにデータを保存
        st.session_state.workflow_data['sensor_data'] = sensor_data
        st.session_state.workflow_data['data_messages'] = data_result['messages_sent']
        
        return f"✅ データ収集完了: {len(data_result['messages_sent'])}件のメッセージ送信"
    
    elif step == 2:
        # ステップ2: プロセス制御エージェント
        process_agent = st.session_state.agents['ProcessControl']
        sensor_data = st.session_state.workflow_data['sensor_data']
        process_result = process_agent.process(sensor_data)
        
        for msg in process_result['messages_sent']:
            st.session_state.communication_log.append(msg)
            st.session_state.total_traffic_kb += msg['size_kb']
        
        st.session_state.workflow_data['process_result'] = process_result
        
        return f"✅ プロセス制御完了: {len(process_result['messages_sent'])}件のメッセージ送信"
    
    elif step == 3:
        # ステップ3: 異常検知エージェント
        anomaly_agent = st.session_state.agents['AnomalyDetection']
        sensor_data = st.session_state.workflow_data['sensor_data']
        anomaly_result = anomaly_agent.process(sensor_data)
        
        for msg in anomaly_result['messages_sent']:
            st.session_state.communication_log.append(msg)
            st.session_state.total_traffic_kb += msg['size_kb']
        
        st.session_state.workflow_data['anomaly_result'] = anomaly_result
        
        anomaly_count = anomaly_result['anomalies']['total_count']
        return f"✅ 異常検知完了: {anomaly_count}件の異常検出、{len(anomaly_result['messages_sent'])}件のメッセージ送信"
    
    elif step == 4:
        # ステップ4: 品質分析エージェント
        quality_agent = st.session_state.agents['QualityAnalysis']
        sensor_data = st.session_state.workflow_data['sensor_data']
        anomaly_result = st.session_state.workflow_data['anomaly_result']
        
        # 3つのラインの平均生産速度を計算
        avg_production_rate = (sensor_data['production_rate_line1'].mean() + 
                               sensor_data['production_rate_line2'].mean() + 
                               sensor_data['production_rate_line3'].mean()) / 3
        
        quality_result = quality_agent.process(
            production_data={'rate': avg_production_rate},
            anomaly_data={'potential_quality_impact': anomaly_result['anomalies']['total_count'] > 3}
        )
        
        for msg in quality_result['messages_sent']:
            st.session_state.communication_log.append(msg)
            st.session_state.total_traffic_kb += msg['size_kb']
        
        st.session_state.workflow_data['quality_result'] = quality_result
        
        quality_score = quality_result['analysis']['quality_score']
        return f"✅ 品質分析完了: 品質スコア {quality_score:.1f}点、{len(quality_result['messages_sent'])}件のメッセージ送信"
    
    elif step == 5:
        # ステップ5: 警告・通知エージェント（異常がある場合のみ）
        anomaly_result = st.session_state.workflow_data['anomaly_result']
        
        if anomaly_result['anomalies']['total_count'] > 0:
            alert_agent = st.session_state.agents['AlertNotification']
            alert_result = alert_agent.process({
                'severity': 'high' if anomaly_result['anomalies']['total_count'] > 5 else 'medium',
                'anomalies': anomaly_result['anomalies'],
                'affected_equipment': ['line_2', 'pump_3']
            })
            
            for msg in alert_result['messages_sent']:
                st.session_state.communication_log.append(msg)
                st.session_state.total_traffic_kb += msg['size_kb']
            
            st.session_state.workflow_data['alert_result'] = alert_result
            
            return f"⚠️ 警告・通知完了: アラートID {alert_result['alert_id']}、{len(alert_result['messages_sent'])}件のメッセージ送信"
        else:
            return "✅ 異常なし: 警告・通知スキップ"
    
    elif step == 6:
        # ステップ6: 予測保全エージェント
        maintenance_agent = st.session_state.agents['PredictiveMaintenance']
        anomaly_result = st.session_state.workflow_data['anomaly_result']
        
        maintenance_result = maintenance_agent.process({
            'vibration_trend': 'increasing' if np.random.random() > 0.5 else 'stable',
            'temperature_instability': anomaly_result['anomalies']['total_count'] > 2
        })
        
        for msg in maintenance_result['messages_sent']:
            st.session_state.communication_log.append(msg)
            st.session_state.total_traffic_kb += msg['size_kb']
        
        st.session_state.workflow_data['maintenance_result'] = maintenance_result
        
        risk_score = maintenance_result['prediction']['risk_score']
        days = maintenance_result['prediction']['days_to_maintenance']
        return f"✅ 予測保全完了: リスクスコア {risk_score:.1f}%、推奨保全まで{days}日、{len(maintenance_result['messages_sent'])}件のメッセージ送信"
    
    elif step == 7:
        # ステップ7: 完了
        return "🎉 全ワークフロー完了！"
    
    return "ステップ実行完了"

# ================================
# UI構築
# ================================

# ヘッダー
st.markdown('<div class="main-header">🤖 AI Multi-Agent Communication System</div>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.header("⚙️ システム制御")
    
    # 🤖 Ollamaステータス表示
    ollama_status = check_ollama_status()
    
    if ollama_status['status'] == 'running' and ollama_status['llama_installed']:
        st.success(f"✅ Ollama稼働中")
        st.caption(f"🦙 Model: {OLLAMA_MODEL}")
    elif ollama_status['status'] == 'running' and not ollama_status['llama_installed']:
        st.warning("⚠️ Ollama稼働中（Llamaモデル未インストール）")
        st.caption(f"実行: `ollama pull {OLLAMA_MODEL}`")
    else:
        st.error("❌ Ollama未起動")
        st.caption("実行: `ollama serve`")
    
    st.markdown("---")
    
    # ワークフローステップ表示
    step_names = [
        "待機中",
        "1️⃣ データ収集",
        "2️⃣ プロセス制御 🤖",
        "3️⃣ 異常検知 🤖",
        "4️⃣ 品質分析 🤖",
        "5️⃣ 警告・通知 🤖",
        "6️⃣ 予測保全 🤖",
        "✅ 完了"
    ]
    
    current_step = st.session_state.workflow_step
    st.markdown(f"### 🧠 Agentの実行")
    st.info(f"{step_names[current_step]}")
    
    # 進行状況バー
    if current_step > 0:
        progress = (current_step - 1) / 6
        st.progress(progress)
        st.caption(f"進捗: {int(progress * 100)}%")
    
    st.markdown("---")
    
    # ワークフロー開始ボタン
    if current_step == 0:
        if st.button("🚀 ワークフロー開始", type="primary", use_container_width=True):
            execute_workflow_step(0)
            st.session_state.workflow_step = 1
            st.session_state.waiting_for_approval = True
            st.rerun()
    
    # 次のステップへ進むボタン
    elif 1 <= current_step <= 6:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ 次へ", type="primary", use_container_width=True):
                result_message = execute_workflow_step(current_step)
                st.session_state.workflow_step += 1
                st.session_state.waiting_for_approval = True
                st.success(result_message)
                st.rerun()
        
        with col2:
            if st.button("⏸️ 一時停止", use_container_width=True):
                st.session_state.waiting_for_approval = True
                st.warning("一時停止中")
    
    # 完了後のリセットボタン
    elif current_step == 7:
        if st.button("🔄 最初から", type="primary", use_container_width=True):
            st.session_state.workflow_step = 0
            st.session_state.workflow_data = {}
            st.session_state.waiting_for_approval = False
            st.rerun()
    
    st.markdown("---")
    
    # 通信ログクリアボタン
    if st.button("🗑️ 通信ログクリア", use_container_width=True):
        st.session_state.communication_log = []
        st.session_state.total_traffic_kb = 0
        st.rerun()
    
    st.markdown("---")
    st.header("📊 通信統計")
    
    st.metric("総通信量", f"{st.session_state.total_traffic_kb:.2f} KB")
    st.metric("メッセージ数", len(st.session_state.communication_log))
    
    if len(st.session_state.communication_log) > 0:
        # 通信量上位のメッセージタイプ
        df_log = pd.DataFrame(st.session_state.communication_log)
        traffic_by_type = df_log.groupby('type')['size_kb'].sum().sort_values(ascending=False)
        
        st.markdown("### 📡 通信量TOP3")
        for i, (msg_type, size) in enumerate(traffic_by_type.head(3).items(), 1):
            traffic_class = "traffic-high" if size > 5 else "traffic-medium" if size > 1 else "traffic-low"
            st.markdown(f"""
            <div class="{traffic_class}">
                {i}. {msg_type}<br>
                {size:.2f} KB
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🤖 エージェント状態")
    
    for name, agent in st.session_state.agents.items():
        status_emoji = "🟢" if agent.status == "completed" else "🟡" if agent.status == "processing" else "⚪"
        st.markdown(f"{status_emoji} **{name}**<br><small>{agent.role}</small>", unsafe_allow_html=True)

# メインエリア - タブ構成
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 通信フロー",
    "💬 メッセージ詳細",
    "📊 通信量分析",
    "💰 AIトークン数"
])

# タブ1: 通信フロー
with tab1:
    st.header("🔄 AIエージェント間通信フロー")
    
    st.markdown("""
    ### 📋 ワークフローの説明
    
    このシステムでは、6つのAIエージェントが**人間の指示なし**に自律的に通信し、
    工場の製造プロセスを最適化します。
    
    **通信の特徴:**
    - ✅ エージェント同士が自動的にメッセージ交換
    - ✅ 優先度に応じた処理順序の自動調整
    - ✅ 異常検出時の自動エスカレーション
    - ✅ フィードバックループによる継続的改善
    """)
    
    # フローチャート
    st.markdown("### 🔀 通信フローチャート")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Mermaid風のフローチャート（テキストベース）
        st.markdown("""
        ```
        データ収集 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ↓ (センサーデータ送信: 大容量)           ↓
            ↓                                        ↓
        プロセス制御 ←→ 異常検知 ←→ 品質分析        ↓
            ↓           ↓           ↓                ↓
            ↓           ↓ (緊急)    ↓                ↓
            ↓           ↓           ↓                ↓
            ↓        警告・通知 ←───┘                ↓
            ↓           ↓                            ↓
            ↓           ↓ (緊急メンテナンス要求)     ↓
            ↓           ↓                            ↓
            └───────→ 予測保全 ←─────────────────────┘
                        ↓
                    (フィードバック)
                        ↓
                    全エージェント
        ```
        """)
    
    with col2:
        st.markdown("""
        **通信量が大きいポイント:**
        
        🔴 **大** (>10KB)
        - データ収集→異常検知
          (生データ転送)
        
        🟡 **中** (1-10KB)
        - 異常検知→警告通知
          (詳細レポート)
        
        🟢 **小** (<1KB)
        - 制御コマンド
        - ステータス更新
        """)
    
    st.markdown("---")
    
    # 最新の通信フロー表示
    if len(st.session_state.communication_log) > 0:
        st.subheader("📨 最新の通信メッセージ (直近10件)")
        
        recent_messages = st.session_state.communication_log[-10:]
        
        for msg in reversed(recent_messages):
            timestamp = msg['timestamp'].strftime('%H:%M:%S')
            size_class = "traffic-high" if msg['size_kb'] > 5 else "traffic-medium" if msg['size_kb'] > 1 else "traffic-low"
            priority_emoji = "🔴" if msg['priority'] == "urgent" else "🟠" if msg['priority'] == "high" else "🟢"
            
            st.markdown(f"""
            <div class="message-flow">
                <div style="flex: 1;">
                    <strong>{msg['from']}</strong> → <strong>{msg['to']}</strong><br>
                    <small>{timestamp} | {msg['type']}</small>
                </div>
                <div style="text-align: right;">
                    {priority_emoji} <span class="{size_class}">{msg['size_kb']:.2f} KB</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# タブ2: メッセージ詳細
with tab2:
    st.header("💬 メッセージ詳細ログ")
    
    if len(st.session_state.communication_log) > 0:
        # フィルター
        col1, col2, col3 = st.columns(3)
        
        with col1:
            unique_senders = list(set([msg['from'] for msg in st.session_state.communication_log]))
            filter_sender = st.selectbox("送信元フィルター", ["すべて"] + unique_senders)
        
        with col2:
            unique_receivers = list(set([msg['to'] for msg in st.session_state.communication_log]))
            filter_receiver = st.selectbox("送信先フィルター", ["すべて"] + unique_receivers)
        
        with col3:
            filter_priority = st.selectbox("優先度フィルター", ["すべて", "urgent", "high", "normal", "low"])
        
        # フィルター適用
        filtered_messages = st.session_state.communication_log
        
        if filter_sender != "すべて":
            filtered_messages = [msg for msg in filtered_messages if msg['from'] == filter_sender]
        
        if filter_receiver != "すべて":
            filtered_messages = [msg for msg in filtered_messages if msg['to'] == filter_receiver]
        
        if filter_priority != "すべて":
            filtered_messages = [msg for msg in filtered_messages if msg['priority'] == filter_priority]
        
        st.markdown(f"### 📊 フィルター結果: {len(filtered_messages)} 件")
        
        # メッセージ詳細表示
        for i, msg in enumerate(reversed(filtered_messages[-20:]), 1):
            with st.expander(f"#{i} | {msg['from']} → {msg['to']} | {msg['type']} | {msg['timestamp'].strftime('%H:%M:%S')}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    **メッセージID:** `{msg['id'][:8]}...`  
                    **優先度:** `{msg['priority']}`  
                    **サイズ:** `{msg['size_kb']:.2f} KB`  
                    **タイムスタンプ:** `{msg['timestamp']}`
                    """)
                
                with col2:
                    st.markdown("**データペイロード:**")
                    st.json(msg['data'])
    else:
        st.info("👆 サイドバーの「ワークフロー実行」ボタンを押してください")

# タブ3: 通信量分析
with tab3:
    st.header("📊 通信量分析")
    
    if len(st.session_state.communication_log) > 0:
        df_log = pd.DataFrame(st.session_state.communication_log)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 送信元別通信量
            traffic_by_sender = df_log.groupby('from')['size_kb'].sum().sort_values(ascending=False)
            
            fig_sender = px.bar(
                x=traffic_by_sender.values,
                y=traffic_by_sender.index,
                orientation='h',
                title="エージェント別送信量",
                labels={'x': '通信量 (KB)', 'y': 'エージェント'},
                color=traffic_by_sender.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig_sender.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_sender, use_container_width=True)
        
        with col2:
            # メッセージタイプ別通信量
            traffic_by_type = df_log.groupby('type')['size_kb'].sum().sort_values(ascending=False)
            
            fig_type = px.pie(
                values=traffic_by_type.values,
                names=traffic_by_type.index,
                title="メッセージタイプ別通信量",
                hole=0.4
            )
            fig_type.update_layout(height=400)
            st.plotly_chart(fig_type, use_container_width=True)
        
        # 通信パターン分析
        st.subheader("🔍 通信パターン分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 優先度別メッセージ数
            priority_counts = df_log['priority'].value_counts()
            
            fig_priority = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="優先度別メッセージ数",
                labels={'x': '優先度', 'y': 'メッセージ数'},
                color=priority_counts.index,
                color_discrete_map={
                    'urgent': '#dc3545',
                    'high': '#ffc107',
                    'normal': '#17a2b8',
                    'low': '#28a745'
                }
            )
            fig_priority.update_layout(height=350)
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col2:
            # 通信ペア分析
            df_log['pair'] = df_log['from'] + ' → ' + df_log['to']
            pair_counts = df_log['pair'].value_counts().head(10)
            
            fig_pair = px.bar(
                x=pair_counts.values,
                y=pair_counts.index,
                orientation='h',
                title="通信ペアTOP10",
                labels={'x': 'メッセージ数', 'y': '通信ペア'}
            )
            fig_pair.update_layout(height=350)
            st.plotly_chart(fig_pair, use_container_width=True)
        
        # 通信量の多いポイント
        st.subheader("🔴 通信量が多いポイント")
        
        heavy_messages = df_log[df_log['size_kb'] > 5].sort_values('size_kb', ascending=False)
        
        if len(heavy_messages) > 0:
            st.dataframe(
                heavy_messages[['from', 'to', 'type', 'size_kb', 'priority', 'timestamp']].head(10),
                use_container_width=True
            )
            
            st.markdown("""
            **大容量通信の理由:**
            - `sensor_data_batch`: センサー生データの一括転送（100データポイント × 5センサー）
            - `anomaly_alert`: 異常検知結果の詳細レポート（タイムスタンプ付き異常リスト）
            - `request_historical_data`: 過去データの要求（30日分の履歴）
            """)
        else:
            st.info("大容量通信（>5KB）は検出されていません")
        
        # 📊 通信量スケーリングシナリオ
        st.subheader("📊 通信量スケーリングシナリオ")
        
        # 現在の通信量
        current_total_kb = df_log['size_kb'].sum()
        current_sensors = 20
        current_datapoints = 1000
        
        # スケーリング係数（データポイント数に比例）
        scaling_scenarios = pd.DataFrame({
            'シナリオ': ['現在', '中規模', '大規模', '超大規模'],
            'センサー数': [20, 50, 100, 500],
            'データポイント': [1000, 5000, 10000, 50000],
            '予想通信量(KB)': [
                current_total_kb,
                current_total_kb * (50/20) * (5000/1000),
                current_total_kb * (100/20) * (10000/1000),
                current_total_kb * (500/20) * (50000/1000)
            ]
        })
        
        # MB単位も追加
        scaling_scenarios['予想通信量(MB)'] = scaling_scenarios['予想通信量(KB)'] / 1024
        
        st.markdown("""
        **仮定:**
        - センサー数とデータポイント数に比例して通信量が増加
        - 現在の構成: 20センサー × 1,000データポイント
        """)
        
        st.dataframe(
            scaling_scenarios[['シナリオ', 'センサー数', 'データポイント', '予想通信量(KB)', '予想通信量(MB)']],
            use_container_width=True,
            hide_index=True
        )
        
        # グラフ化
        fig_scaling = px.bar(
            scaling_scenarios,
            x='シナリオ',
            y='予想通信量(MB)',
            title='スケーリングシナリオ別の予想通信量',
            text='予想通信量(MB)',
            color='予想通信量(MB)',
            color_continuous_scale='Blues'
        )
        
        fig_scaling.update_traces(texttemplate='%{text:.2f} MB', textposition='outside')
        fig_scaling.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig_scaling, use_container_width=True)
    
    else:
        st.info("👆 サイドバーの「ワークフロー実行」ボタンを押してください")

# タブ4: AIトークン数
with tab4:
    st.header("💰 AIエージェント トークン数 & コスト分析")
    
    st.markdown("""
    ### 📊 概要
    
    AIエージェントが扱うIoTデータ量が増加すると、エージェント間の通信（クエリ）も爆発的に増加します。
    ここでは、各AIモデルでの**トークン消費量**と**料金**を可視化します。
    
    **分析対象AI（フラッグシップモデル）:**
    - 🟢 **OpenAI GPT-4o**
    - 🔵 **Anthropic Claude Sonnet 4**
    - 🔴 **Google Gemini 1.5 Pro**
    """)
    
    if len(st.session_state.communication_log) > 0:
        df_log = pd.DataFrame(st.session_state.communication_log)
        
        # トークン統計
        total_tokens = df_log['tokens'].sum()
        total_cost_gpt4o = df_log['cost_gpt4o'].sum()
        total_cost_claude_sonnet = df_log['cost_claude_sonnet'].sum()
        total_cost_gemini_pro = df_log['cost_gemini_pro'].sum()
        
        # 円換算
        total_cost_gpt4o_jpy = total_cost_gpt4o * USD_TO_JPY
        total_cost_claude_sonnet_jpy = total_cost_claude_sonnet * USD_TO_JPY
        total_cost_gemini_pro_jpy = total_cost_gemini_pro * USD_TO_JPY
        
        # サマリーメトリクス
        st.subheader("📈 総トークン数とコスト")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("総トークン数", f"{total_tokens:,}")
            st.caption("全メッセージの合計")
        
        with col2:
            st.metric("GPT-4o", f"¥{total_cost_gpt4o_jpy:.2f}")
            st.caption("OpenAI フラッグシップ")
        
        with col3:
            st.metric("Claude Sonnet 4", f"¥{total_cost_claude_sonnet_jpy:.2f}")
            st.caption("Anthropic フラッグシップ")
        
        with col4:
            st.metric("Gemini 1.5 Pro", f"¥{total_cost_gemini_pro_jpy:.2f}")
            st.caption("Google フラッグシップ")
        
        st.markdown("---")
        
        # エージェント別トークン数
        st.subheader("🤖 エージェント別トークン消費")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 送信元別トークン数
            tokens_by_sender = df_log.groupby('from')['tokens'].sum().sort_values(ascending=False)
            
            fig_tokens_sender = go.Figure()
            fig_tokens_sender.add_trace(go.Bar(
                x=tokens_by_sender.values,
                y=tokens_by_sender.index,
                orientation='h',
                marker=dict(
                    color=tokens_by_sender.values,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="トークン数")
                ),
                text=tokens_by_sender.values,
                texttemplate='%{text:,}',
                textposition='outside'
            ))
            
            fig_tokens_sender.update_layout(
                title="エージェント別送信トークン数",
                xaxis_title="トークン数",
                yaxis_title="エージェント",
                height=400
            )
            
            st.plotly_chart(fig_tokens_sender, use_container_width=True)
        
        with col2:
            # コスト比較（3モデル）- 円換算
            cost_comparison = pd.DataFrame({
                'モデル': ['GPT-4o', 'Claude Sonnet 4', 'Gemini 1.5 Pro'],
                'コスト(JPY)': [
                    total_cost_gpt4o_jpy,
                    total_cost_claude_sonnet_jpy,
                    total_cost_gemini_pro_jpy
                ]
            })
            
            fig_cost_comparison = px.bar(
                cost_comparison,
                x='モデル',
                y='コスト(JPY)',
                title="AIモデル別コスト比較（フラッグシップ）",
                color='コスト(JPY)',
                color_continuous_scale='RdYlGn_r',
                text='コスト(JPY)'
            )
            
            fig_cost_comparison.update_traces(texttemplate='¥%{text:.2f}', textposition='outside')
            fig_cost_comparison.update_layout(height=400)
            st.plotly_chart(fig_cost_comparison, use_container_width=True)
        
        # トークン数の時系列推移
        st.subheader("📉 トークン消費の時系列推移")
        
        df_log['cumulative_tokens'] = df_log['tokens'].cumsum()
        
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log['cumulative_tokens'],
            mode='lines',
            name='累積トークン数',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig_timeline.update_layout(
            title="累積トークン数の推移（メッセージごと）",
            xaxis_title="メッセージ番号",
            yaxis_title="累積トークン数",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # 💥 データ量爆増の可視化
        st.subheader("💥 IoTデータ量 vs トークン消費の爆増")
        
        st.markdown("""
        **重要な発見:**
        - 📊 1,000データポイント × 20センサー = **20,000データポイント**
        - 💬 これを異常検知エージェントに送信すると、**数万〜数十万トークン**を消費
        - 💰 大規模IoTシステムでは、**月額数百〜数千ドル**のAIコストが発生する可能性
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # メッセージタイプ別トークン数
            tokens_by_type = df_log.groupby('type')['tokens'].sum().sort_values(ascending=False).head(10)
            
            fig_tokens_type = px.pie(
                values=tokens_by_type.values,
                names=tokens_by_type.index,
                title="メッセージタイプ別トークン分布",
                hole=0.4
            )
            
            fig_tokens_type.update_layout(height=400)
            st.plotly_chart(fig_tokens_type, use_container_width=True)
        
        with col2:
            # 高トークンメッセージTOP5
            top_token_messages = df_log.nlargest(5, 'tokens')[['from', 'to', 'type', 'tokens', 'size_kb']]
            
            st.markdown("### 🔥 高トークンメッセージ TOP5")
            
            for idx, row in top_token_messages.iterrows():
                st.markdown(f"""
                <div style="background-color: #fff3cd; padding: 1rem; border-radius: 5px; 
                            margin-bottom: 0.5rem; border-left: 4px solid #ffc107;">
                    <strong>{row['from']} → {row['to']}</strong><br>
                    <small>{row['type']}</small><br>
                    💬 <strong>{row['tokens']:,} tokens</strong> | 📦 {row['size_kb']:.2f} KB
                </div>
                """, unsafe_allow_html=True)
        
        # コスト詳細テーブル - 円換算
        st.subheader("💵 詳細コストテーブル")
        
        cost_table = df_log[['from', 'to', 'type', 'tokens', 'size_kb', 
                             'cost_gpt4o', 'cost_claude_sonnet', 'cost_gemini_pro']].copy()
        
        # 円換算
        cost_table['cost_gpt4o_jpy'] = cost_table['cost_gpt4o'] * USD_TO_JPY
        cost_table['cost_claude_sonnet_jpy'] = cost_table['cost_claude_sonnet'] * USD_TO_JPY
        cost_table['cost_gemini_pro_jpy'] = cost_table['cost_gemini_pro'] * USD_TO_JPY
        
        cost_table = cost_table[['from', 'to', 'type', 'tokens', 'size_kb', 
                                 'cost_gpt4o_jpy', 'cost_claude_sonnet_jpy', 'cost_gemini_pro_jpy']]
        cost_table.columns = ['送信元', '送信先', 'タイプ', 'トークン数', 'サイズ(KB)', 
                              'GPT-4o(JPY)', 'Claude Sonnet(JPY)', 'Gemini Pro(JPY)']
        
        st.dataframe(
            cost_table.tail(20),
            use_container_width=True,
            height=400
        )
        
        # 月額コスト予測 - 円換算
        st.subheader("📅 月額コスト予測")
        
        st.markdown("""
        **仮定:**
        - このワークフローを **1時間に1回** 実行
        - **24時間 × 30日** = 月720回実行
        - **為替レート: $1 = ¥155**
        """)
        
        monthly_multiplier = 720
        
        cost_projection = pd.DataFrame({
            'モデル': ['GPT-4o', 'Claude Sonnet 4', 'Gemini 1.5 Pro'],
            '1回あたり(JPY)': [
                total_cost_gpt4o_jpy,
                total_cost_claude_sonnet_jpy,
                total_cost_gemini_pro_jpy
            ],
            '月額予測(JPY)': [
                total_cost_gpt4o_jpy * monthly_multiplier,
                total_cost_claude_sonnet_jpy * monthly_multiplier,
                total_cost_gemini_pro_jpy * monthly_multiplier
            ]
        })
        
        fig_monthly = go.Figure()
        
        fig_monthly.add_trace(go.Bar(
            x=cost_projection['モデル'],
            y=cost_projection['月額予測(JPY)'],
            text=cost_projection['月額予測(JPY)'].apply(lambda x: f'¥{x:.0f}'),
            textposition='outside',
            marker=dict(
                color=['#10a37f', '#b575e3', '#4285f4'],  # OpenAI Green, Claude Purple, Google Blue
                line=dict(color='white', width=2)
            )
        ))
        
        fig_monthly.update_layout(
            title="月額コスト予測（月720回実行時）",
            xaxis_title="AIモデル",
            yaxis_title="月額コスト (JPY)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # 最もコスト効率の良いモデル - 円換算
        min_cost_model = cost_projection.loc[cost_projection['月額予測(JPY)'].idxmin(), 'モデル']
        min_cost_value = cost_projection['月額予測(JPY)'].min()
        max_cost_model = cost_projection.loc[cost_projection['月額予測(JPY)'].idxmax(), 'モデル']
        max_cost_value = cost_projection['月額予測(JPY)'].max()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            ✅ **最もコスト効率が良い:** {min_cost_model}
            
            月額予測: ¥{min_cost_value:.0f}
            """)
        
        with col2:
            st.error(f"""
            ⚠️ **最もコストが高い:** {max_cost_model}
            
            月額予測: ¥{max_cost_value:.0f}
            
            差額: ¥{max_cost_value - min_cost_value:.0f} ({((max_cost_value / min_cost_value - 1) * 100):.1f}%増)
            """)
        
        # 重要なインサイト
        st.subheader("💡 重要なインサイト")
        
        st.markdown(f"""
        ### 🔥 データ量増加によるコスト爆増
        
        現在の設定:
        - **センサー数:** 20個
        - **データポイント:** 1,000個
        - **総データポイント:** 20,000個
        - **総トークン数:** {total_tokens:,}
        - **GPT-4oでの1回コスト:** ¥{total_cost_gpt4o_jpy:.2f}
        
        ### 📈 スケーリングシナリオ
        
        | シナリオ | センサー数 | データポイント | 予想トークン数 | GPT-4o 月額コスト |
        |---------|----------|--------------|--------------|------------------|
        | 現在 | 20 | 1,000 | {total_tokens:,} | ¥{total_cost_gpt4o_jpy * 720:.0f} |
        | 中規模 | 50 | 5,000 | {total_tokens * 12:,} | ¥{total_cost_gpt4o_jpy * 720 * 12:.0f} |
        | 大規模 | 100 | 10,000 | {total_tokens * 50:,} | ¥{total_cost_gpt4o_jpy * 720 * 50:.0f} |
        | 超大規模 | 500 | 50,000 | {total_tokens * 1250:,} | ¥{total_cost_gpt4o_jpy * 720 * 1250:.0f} |
        
        ### ⚠️ 結論
        
        - IoTデータ量が**10倍**になると、AIコストも**約10倍**に増加
        - 大規模IoTシステムでは、**コスト効率の良いAIモデル選択**が極めて重要
        - **為替レート: $1 = ¥155で計算**
        - **フラッグシップモデル間の比較:**
          - 🥇 **Gemini 1.5 Pro**: 最もコスト効率が良い（入力¥193.75/1M tokens）
          - 🥈 **GPT-4o**: バランス型（入力¥387.50/1M tokens）
          - 🥉 **Claude Sonnet 4**: 最高品質だが高コスト（入力¥465.00/1M tokens）
        """)
        
    else:
        st.info("👆 サイドバーの「ワークフロー開始」→「次へ」でデータを生成してください")

# フッター
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**システムバージョン**")
    st.info("📦 v3.0.0 - AI Communication")

with col2:
    st.markdown("**最終更新**")
    st.info(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.markdown("**通信プロトコル**")
    st.success("✅ Agent-to-Agent Protocol")
