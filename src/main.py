from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import traceback
import logging
from datetime import datetime
import json
from pathlib import Path
from bc_csv7 import RecommendationRequest
import uvicorn
from fastapi.middleware.cors import CORSMiddleware



# 導入原本的推薦系統
from bc_csv7 import (
    initialize_system, 
    analyze_user_intent, 
    recommend_advanced,
    calculate_sensor_type_similarity,
    calculate_module_similarity,
    calculate_environment_similarity
)

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 創建 FastAPI 應用
app = FastAPI(
    title="智慧感測器推薦系統",
    description="基於 AI 的感測器推薦 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 測試階段先開放全部，正式環境可指定來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 靜態文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 全域變數儲存系統狀態
system_state = {
    "df": None,
    "model": None,
    "device_embeddings": None,
    "initialized": False,
    "error_message": None
}

# Pydantic 模型定義

class SensorRecommendation(BaseModel):
    name: str
    type: str
    final_score: float
    sensor_type_similarity: float
    module_similarity: float
    semantic_similarity: float
    environment_similarity: float
    compatible_modules: List[str] = []
    features: Optional[str] = None
    ip_rating: Optional[str] = None
    power_consumption: Optional[float] = None
    operating_temp: Optional[str] = None
    range: Optional[str] = None
    precision: Optional[str] = None

class IntentAnalysis(BaseModel):
    direct_sensor_needs: List[str] = []
    environmental_context: List[str] = []
    exclude_keywords: List[str] = []
    primary_application: Optional[str] = None
    environment_needs: List[str] = []
    technical_specs: List[str] = []

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    intent_analysis: Optional[IntentAnalysis] = None
    recommendations: List[SensorRecommendation] = []
    total_found: int = 0
    search_timestamp: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("驗證錯誤：", exc)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

# 系統初始化
@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化系統"""
    try:
        logger.info("正在初始化感測器推薦系統...")
        df, model, device_embeddings = initialize_system("sensors.csv")
        
        if df is not None and model is not None and device_embeddings is not None:
            system_state["df"] = df
            system_state["model"] = model
            system_state["device_embeddings"] = device_embeddings
            system_state["initialized"] = True
            logger.info(f"系統初始化成功，載入 {len(df)} 筆感測器資料")
        else:
            system_state["error_message"] = "系統初始化失敗：無法載入感測器資料"
            logger.error(system_state["error_message"])
    except Exception as e:
        system_state["error_message"] = f"系統初始化失敗：{str(e)}"
        logger.error(f"系統初始化失敗：{e}")
        logger.error(traceback.format_exc())

# 首頁路由
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """返回主頁面"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "system_ready": system_state["initialized"]
    })

# 系統狀態檢查
@app.get("/api/status")
async def get_system_status():
    """檢查系統狀態"""
    return {
        "initialized": system_state["initialized"],
        "error_message": system_state["error_message"],
        "total_sensors": len(system_state["df"]) if system_state["df"] is not None else 0
    }

# 主要推薦 API

@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_sensors(request: RecommendationRequest):
    """感測器推薦 API"""
    
    # 檢查系統是否已初始化
    if not system_state["initialized"]:
        raise HTTPException(
            status_code=503, 
            detail=f"系統尚未就緒：{system_state['error_message']}"
        )
    
    try:
        # 分析使用者意圖
        intent = analyze_user_intent(request.query)
        
        # 執行推薦
        result = recommend_advanced(
            request.query,
            system_state["df"],
            system_state["model"],
            system_state["device_embeddings"],
            sensor_type_weight=request.sensor_type_weight,
            module_weight=request.module_weight,
            semantic_weight=request.semantic_weight,
            environment_weight=request.environment_weight,
            threshold=request.threshold,
            top_k=request.top_k
        )
        
        # 處理推薦結果
        recommendations = []
        total_found = 0
        
        if result is not None and not result.empty:
            total_found = len(result)
            
            for _, row in result.iterrows():
                # 安全地獲取值
                def safe_get(key, default=None):
                    try:
                        val = row.get(key, default)
                        if pd.isna(val):
                            return default
                        return val
                    except:
                        return default
                
                # 處理相容模組
                modules = safe_get('parsed_modules', [])
                if not isinstance(modules, list):
                    modules = []
                
                recommendation = SensorRecommendation(
                    name=safe_get('name', '未知'),
                    type=safe_get('type', '未分類'),
                    final_score=round(safe_get('final_score', 0.0), 3),
                    sensor_type_similarity=round(safe_get('sensor_type_similarity', 0.0), 3),
                    module_similarity=round(safe_get('module_similarity', 0.0), 3),
                    semantic_similarity=round(safe_get('semantic_similarity', 0.0), 3),
                    environment_similarity=round(safe_get('environment_similarity', 0.0), 3),
                    compatible_modules=modules,
                    features=safe_get('features'),
                    ip_rating=safe_get('ip_rating'),
                    power_consumption=safe_get('power_consumption'),
                    operating_temp=safe_get('operating_temp'),
                    range=safe_get('range'),
                    precision=safe_get('precision')
                )
                recommendations.append(recommendation)
        
        print("使用者輸入：", request.query)
        print("門檻值與推薦數量：", request.threshold, request.top_k)

        # 建立意圖分析結果
        intent_analysis = IntentAnalysis(
            direct_sensor_needs=intent.get('direct_sensor_needs', []),
            environmental_context=intent.get('environmental_context', []),
            exclude_keywords=intent.get('exclude_keywords', []),
            primary_application=intent.get('primary_application'),
            environment_needs=intent.get('environment_needs', []),
            technical_specs=intent.get('technical_specs', [])
        )
        
        # 生成回應訊息
        if total_found > 0:
            message = f"成功找到 {total_found} 款推薦感測器"
        else:
            message = "很抱歉，沒有找到符合需求的感測器，請嘗試調整搜尋條件"
        
        return RecommendationResponse(
            success=True,
            message=message,
            intent_analysis=intent_analysis,
            recommendations=recommendations,
            total_found=total_found,
            search_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"推薦過程發生錯誤：{e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"推薦過程發生錯誤：{str(e)}")

# 快速搜尋 API（簡化版）
@app.get("/api/quick-search")
async def quick_search(q: str, limit: int = 5):
    """快速搜尋感測器"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="系統尚未就緒")
    
    try:
        # 使用預設參數進行快速搜尋
        result = recommend_advanced(
            q,
            system_state["df"],
            system_state["model"],
            system_state["device_embeddings"],
            threshold=0.3,
            top_k=limit
        )
        
        if result is None or result.empty:
            return {"results": [], "total": 0}
        
        # 簡化回應格式
        results = []
        for _, row in result.iterrows():
            results.append({
                "name": row.get('name', '未知'),
                "type": row.get('type', '未分類'),
                "score": round(row.get('final_score', 0.0), 3),
                "features": row.get('features', ''),
                "modules": row.get('parsed_modules', [])
            })
        
        return {"results": results, "total": len(results)}
        
    except Exception as e:
        logger.error(f"快速搜尋錯誤：{e}")
        raise HTTPException(status_code=500, detail=f"搜尋失敗：{str(e)}")

# 獲取感測器類型統計
@app.get("/api/sensor-types")
async def get_sensor_types():
    """獲取感測器類型統計"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="系統尚未就緒")
    
    try:
        df = system_state["df"]
        type_counts = df['type'].value_counts().to_dict()
        
        return {
            "sensor_types": type_counts,
            "total_sensors": len(df)
        }
        
    except Exception as e:
        logger.error(f"獲取感測器類型統計錯誤：{e}")
        raise HTTPException(status_code=500, detail=f"獲取統計失敗：{str(e)}")

# 健康檢查
@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "system_ready": system_state["initialized"]
    }

# 錯誤處理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 異常處理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """一般異常處理器"""
    logger.error(f"未處理的異常：{exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "內部伺服器錯誤，請稍後再試",
            "timestamp": datetime.now().isoformat()
        }
    )

# 主程式入口
if __name__ == "__main__":
    
    # 開發模式配置
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
