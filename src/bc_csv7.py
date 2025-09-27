import pandas as pd
import re
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import ast
from difflib import get_close_matches
from pydantic import BaseModel, Field

# 初始化
def initialize_system(csv_file="sensors.csv"):
    try:
        df = pd.read_csv(csv_file)
        print(f"載入 {len(df)} 筆感測器資料")
        
        # 處理 compatible_modules 欄位
        df["compatible_modules"] = df["compatible_modules"].fillna("")
        df["parsed_modules"] = df["compatible_modules"].apply(parse_compatible_modules)
        
        # 增強text
        df["search_text"] = df.apply(lambda row: create_enhanced_search_text(row), axis=1)

        print("載入語意模型...")
        model = SentenceTransformer("./model")

        print("建立語意向量...")
        device_embeddings = model.encode(df["search_text"].tolist(), convert_to_tensor=True)
        
        print("系統初始化完成！")
        return df, model, device_embeddings
        
    except Exception as e:
        print(f"初始化錯誤：{e}")
        return None, None, None

def parse_compatible_modules(modules_str):
    if not modules_str or modules_str == "":
        return []
    
    try:
        cleaned = str(modules_str).strip('{}').replace('"', '').replace("'", "")
        if not cleaned:
            return []
        modules = [module.strip() for module in cleaned.split(',')]
        return [module for module in modules if module]
    except Exception as e:
        return []

def create_enhanced_search_text(row):
    text_parts = []
    
    # 感測器名稱和類型 (高權重)
    if pd.notna(row.get('name')):
        name = str(row['name'])
        text_parts.extend([name] * 5)
    
    if pd.notna(row.get('type')):
        sensor_type = str(row['type'])
        text_parts.extend([sensor_type] * 4)
    
    # 相容模組 (最高權重)
    if 'parsed_modules' in row.index and row['parsed_modules']:
        for module in row['parsed_modules']:
            text_parts.extend([module] * 4)
    
    # 特徵描述
    if pd.notna(row.get('features')):
        features = str(row['features'])
        text_parts.append(features)
        # 提取環境和應用關鍵字
        keywords = extract_application_keywords(features)
        text_parts.extend(keywords)
    
    # 環境適用性資訊
    env_info = extract_environmental_suitability(row)
    text_parts.extend(env_info)
    
    return " ".join(text_parts)

def extract_application_keywords(features_text):
    keywords = []

    application_patterns = {
        '室內監控': [r'室內', r'建築', r'辦公', r'機房'],
        '安全監控': [r'監控', r'安全', r'預警', r'警報'],
        '熱源偵測': [r'熱源', r'熱顯像', r'紅外線', r'溫度'],
        '人員偵測': [r'人員', r'人像', r'人流', r'體溫'],
        '火災預防': [r'火災', r'火源', r'煙霧', r'預警'],
        '環境監測': [r'環境', r'氣候', r'空氣', r'品質'],
        '工業應用': [r'工廠', r'工業', r'機械', r'設備'],
        '農業應用': [r'農業', r'溫室', r'土壤', r'種植'],
        '戶外應用': [r'戶外', r'森林', r'野外', r'氣象']
    }
    
    for app_type, patterns in application_patterns.items():
        if any(re.search(pattern, features_text) for pattern in patterns):
            keywords.extend([app_type] * 2)
    
    return keywords

def extract_environmental_suitability(row):
    env_info = []
    
    # IP等級
    if pd.notna(row.get('ip_rating')):
        ip_rating = str(row['ip_rating']).upper()
        if 'IP65' in ip_rating or 'IP66' in ip_rating:
            env_info.extend(['防塵防水', '室外適用', '惡劣環境'] * 2)
        elif 'IPX7' in ip_rating or 'IPX8' in ip_rating:
            env_info.extend(['防水', '戶外可用'])
        elif ip_rating != '未標示' and ip_rating != '未指定':
            env_info.append('環境保護')
    
    # 工作溫度
    if pd.notna(row.get('operating_temp')):
        temp_range = str(row['operating_temp'])
        if '-' in temp_range:
            try:
                temps = re.findall(r'-?\d+', temp_range)
                if len(temps) >= 2:
                    min_temp = int(temps[0])
                    max_temp = int(temps[1])
                    
                    if min_temp <= -20:
                        env_info.extend(['極低溫', '嚴寒環境'])
                    if max_temp >= 85:
                        env_info.extend(['高溫', '工業環境'])
                    if min_temp >= 0 and max_temp <= 50:
                        env_info.extend(['室內環境', '一般環境'])
                    else:
                        env_info.extend(['寬溫範圍', '惡劣環境'])
            except:
                pass
    
    # 功耗
    if pd.notna(row.get('power_consumption')):
        try:
            power = float(row['power_consumption'])
            if power <= 0.01:
                env_info.extend(['超低功耗', '電池供電'])
            elif power <= 0.1:
                env_info.extend(['低功耗', '節能'])
        except:
            pass
    
    return env_info

def analyze_user_intent(user_input: str):
    """分析使用者的具體需求意圖，區分直接需求和環境描述"""
    user_lower = user_input.lower()
    comprehensive_analysis = {
        # intent
        'direct_sensor_needs': [],      # 直接的感測器需求
        'environmental_context': [],    # 環境背景描述
        'exclude_keywords': [],         # 需要排除的關鍵字
        
        # needs
        'primary_application': None,    # 主要應用場景
        'environment_needs': [],        # 環境適應需求
        'technical_specs': [],          # 技術規格需求
        'priority_features': [],        # 優先功能特性

        'application_domain': None,     # 應用領域
        'deployment_context': [],       # 部署環境
        'performance_requirements': [] # 效能需求
    }

    # 直接感測器需求識別
    direct_needs_patterns = {
        '熱顯像': [r'紅外線.*熱顯像', r'熱顯像.*模組', r'熱顯像', r'熱像儀', r'體溫.*感測', r'溫度異常', r'紅外線.*影像'],
        '毫米波雷達': [r'毫米波.*人流', r'毫米波.*模組', r'人流.*計算', r'人員.*追蹤', r'動線.*監控', r'人體.*偵測', r'毫米波雷達'],
        '氣體感測': [r'氣體.*感測', r'氣體.*偵測', r'氣體.*檢測', r'co2.*感測', r'co₂.*感測', r'voc.*感測', r'空氣品質.*感測', r'臭氧.*感測', r'一氧化碳.*感測', r'氨氣.*感測', r'可燃氣體.*感測'],
        '二氧化碳氣體感測': [r'二氧化碳.*感測', r'co2.*感測',r'co2', r'二氧化碳', r'co₂.*濃度', r'co2濃度', r'co2.*檢測', r'co₂(感測器|模組|設備)'],
        '溫濕度': [r'溫濕度.*感測', r'溫度.*濕度.*監控',  r'溫濕度',r'環境.*溫濕度', r'溫度.*感測', r'濕度.*感測', r'室內.*溫度'],
        '環境光感測': [r'光照.*感測', r'照度.*偵測', r'亮度.*監控', r'環境光.*感測'],
        '傾斜/振動': [r'傾斜.*偵測', r'振動.*感測', r'角度.*感測', r'晃動.*感測', r'地震.*預警', r'結構.*監控',r'傾斜.*(感測器|模組|設備)?'],
        '超音波風速風向': [r'風速.*感測', r'風向.*感測', r'風力.*偵測', r'風速風向', r'氣象.*監控', r'超音波.*風速']
    }

    
    environmental_context_patterns = {
        '低溫環境': {
            'patterns': [r'低溫.*環境', r'冷藏.*倉庫', r'冷凍.*環境', r'極低溫'],
            'requirement': '低溫環境適用'
        },
        '高溫環境': {
            'patterns': [r'高溫.*環境', r'極高溫'],
            'requirement': '高溫環境適用'
        },
        '高濕環境': {
            'patterns': [r'高濕.*環境', r'潮濕.*環境', r'濕度.*較高'],
            'requirement': '抗濕環境'
        },
        '工業環境': {
            'patterns': [r'工廠.*環境', r'工業.*現場', r'生產.*環境'],
            'requirement': '工業級耐用性'
        },
        '室內環境': {
            'patterns': [r'室內.*環境', r'建築.*內部', r'密閉.*空間'],
            'requirement': '室內部署適用'
        },
        '戶外環境': {
            'patterns': [r'戶外.*環境', r'野外.*應用', r'室外.*監控'],
            'requirement': '戶外防護等級'
        }
    }
    
    technical_specs_patterns = {
        'AI功能': {
            'patterns': [r'AI', r'人工智慧', r'機器學習', r'行為辨識', r'智能分析'],
            'spec': 'AI擴充相容性'
        },
        '即時監控': {
            'patterns': [r'即時', r'實時', r'連續監測', r'持續監控'],
            'spec': '連續監控能力'
        },
        '無線傳輸': {
            'patterns': [r'無線', r'wifi', r'藍牙', r'zigbee', r'lora'],
            'spec': '無線通訊功能'
        },
        '低功耗': {
            'patterns': [r'低功耗', r'省電', r'電池供電', r'節能'],
            'spec': '低功耗設計'
        },
        '高精度': {
            'patterns': [r'高精度', r'精確', r'準確度', r'精密'],
            'spec': '高精度測量'
        }
    }
  # === 應用領域識別 ===
    application_domain_patterns = {
        '智慧農業': [r'農業', r'溫室', r'種植', r'農作物', r'灌溉'],
        '工業監控': [r'工廠', r'生產線', r'機械', r'設備監控', r'預測維護'],
        '建築安全': [r'建築', r'結構', r'安全監控', r'火災預警', r'安防'],
        '環境監測': [r'環境', r'氣候', r'空氣品質', r'污染監測', r'氣象'],
        '智慧城市': [r'城市', r'交通', r'公共設施', r'智慧路燈', r'人流統計']
    }

    
    # 1. 直接感測器需求
    for need_type, patterns in direct_needs_patterns.items():
        if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in patterns):
            comprehensive_analysis['direct_sensor_needs'].append(need_type)

    # 2. 環境背景
    for env_type, config in environmental_context_patterns.items():
        if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in config['patterns']):
            comprehensive_analysis['environmental_context'].append(env_type)
            comprehensive_analysis['environment_needs'].append(config['requirement'])

    # 3. 技術規格需求
    for tech_type, config in technical_specs_patterns.items():
        if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in config['patterns']):
            comprehensive_analysis['technical_specs'].append(config['spec'])

    # 4. 應用領域
    for domain, patterns in application_domain_patterns.items():
        if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in patterns):
            comprehensive_analysis['application_domain'] = domain
            break
    # exclude_keywords
    if any(ctx in comprehensive_analysis['environmental_context'] for ctx in ['低溫環境', '高濕環境','高溫環境']):
        if '溫濕度' not in comprehensive_analysis['direct_sensor_needs']:
            comprehensive_analysis['exclude_keywords'].append('溫濕度')

    return comprehensive_analysis
    
# 感測器類型匹配度計算

def calculate_sensor_type_similarity(user_input: str, df: pd.DataFrame):
    intent = analyze_user_intent(user_input)
    similarities = []
    user_lower = user_input.lower()

    # 感測器類型關鍵字
    sensor_type_keywords = {
        '熱顯像': {
            'primary': ['熱顯像', '紅外線', '熱源', '火源', '體溫', '溫度影像', '人員追蹤', '熱成像'],
            'secondary': ['無接觸', '監控'],
            # 'exclude': ['傾斜', '振動']
        },
        '溫濕度': {
            'primary': ['溫度', '濕度', '溫濕度', '環境溫度', '環境濕度','空氣濕潤度'],
            'secondary': ['監控', '控制', '農業', '環境', '溫室', '機房', '倉儲']
        },

        '氣體感測': {
            'primary': ['氨氣', '氣體', '空氣品質', '一氧化碳', '二氧化碳','co2', 'co₂', "甲烷", "VOC", "氣體檢測", "氣體洩漏",
                        '可燃氣體','空氣品質監控', '室內空氣', '通風','空氣監測', '空氣檢測','排風','氣體濃度'],
            'secondary': ['工業監控', '氣體洩漏'],
        },
        '毫米波雷達': {
            'primary': ['毫米波', '雷達', '人流', '人數統計', '動線追蹤'],
            'secondary': ['人員偵測', '流量統計'],
        },
        '超音波風速風向': {
            'primary': ['風速', '風向', '氣象', '風力'],
            'secondary': ['氣象', '氣候'],
        },
        '環境光感測': {
            'primary': ['光照', '照度', '亮度', '環境光', '光度','光照度','日照強度'],
            'secondary': ['農業', '日照', '光源'],
        },
        '傾斜/振動': {
            'primary': ['傾斜', '振動', '角度', '穩定性監測', '機械振動', '結構監測'],
            'secondary': ['機械監測', '設備監控'],
        },        
        '二氧化碳氣體感測': {
            'primary': ['CO2', 'CO₂', '二氧化碳', '空氣品質', '空氣監測', 'co2', 'co₂',  '二氧化碳濃度', 
            'CO2濃度', "ppm",'火災'],
            'secondary': ['通風', '空調', '室內環境'],
        },
        
    }

    for _, row in df.iterrows():
        max_similarity = 0.0
        sensor_type = str(row.get('type', '')).strip()

        # 排除指定關鍵字
        if any(ex in sensor_type for ex in intent['exclude_keywords']):
            similarities.append(0.0)
            continue

        # 直接需求給高分
        if sensor_type in intent['direct_sensor_needs']:
            similarities.append(0.9)
            continue

        # 根據關鍵字計算匹配度
        if sensor_type in sensor_type_keywords:
            kws = sensor_type_keywords[sensor_type]
            primary_matches = sum(1 for kw in kws['primary'] if kw in user_lower)
            secondary_matches = sum(1 for kw in kws['secondary'] if kw in user_lower)
            if primary_matches:
                if len(kws['primary'])>12:
                    primary_score = min(primary_matches / (len(kws['primary'])/2.5) * 4, 1.0)
                    secondary_score = min(secondary_matches / (len(kws['secondary'])/1.3) * 0.3, 0.2)
                elif len(kws['primary'])>5:
                    primary_score = min(primary_matches / (len(kws['primary'])/1.7) * 4, 1.0)
                    secondary_score = min(secondary_matches / (len(kws['secondary'])/1.3) * 0.3, 0.2)
                else:
                    primary_score = min(primary_matches / (len(kws['primary'])/1.4) * 4, 1.0)
                    secondary_score = min(secondary_matches / (len(kws['secondary'])/1) * 0.3, 0.2)
                max_similarity = min(primary_score + secondary_score, 1.0)

        similarities.append(max_similarity)

    return np.array(similarities)

#推薦邏輯
def recommend_advanced(user_input: str, df, model, device_embeddings, 
                        sensor_type_weight: float = 0.4,    # 類型權重
                        module_weight: float = 0.3,         # 模組權重
                        semantic_weight: float = 0.25,       # 語意權重
                        environment_weight: float = 0.05,    # 環境權重
                        threshold:float = 0.5,            
                        top_k:int= 3):
    
    if df is None or model is None or device_embeddings is None:
        return None
    
    try:
        # 1. 語意相似度計算
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        semantic_similarities = util.cos_sim(user_embedding, device_embeddings)[0].cpu().numpy()
        
        # 2. 感測器類型匹配度
        sensor_type_similarities = calculate_sensor_type_similarity(user_input, df)
        
        # 3. 模組匹配度
        module_similarities = calculate_module_similarity(user_input, df)
        
        # 4. 環境適用性匹配度
        environment_similarities = calculate_environment_similarity(user_input, df)
        
        # 5. 綜合評分
        final_scores = (sensor_type_similarities * sensor_type_weight + 
                       module_similarities * module_weight + 
                       semantic_similarities * semantic_weight +
                       environment_similarities * environment_weight)
        
        # 6. 建立結果
        df_copy = df.copy()
        df_copy["sensor_type_similarity"] = sensor_type_similarities
        df_copy["module_similarity"] = module_similarities
        df_copy["semantic_similarity"] = semantic_similarities
        df_copy["environment_similarity"] = environment_similarities
        df_copy["final_score"] = final_scores
        
        # 7. 篩選和排序
        matched = df_copy[df_copy["final_score"] >= threshold].sort_values(
            by="final_score", ascending=False)
        
        if matched.empty:
            return None
        
        # 8. 選擇顯示欄位
        display_columns = ['name', 'type', 'final_score', 'sensor_type_similarity',
                          'module_similarity', 'semantic_similarity', 'environment_similarity',
                          'parsed_modules', 'features', 'ip_rating', 'power_consumption', 
                          'operating_temp', 'range', 'precision']
        
        # 只保留存在的欄位
        display_columns = [col for col in display_columns if col in matched.columns]
        
        result = matched[display_columns].head(top_k).reset_index(drop=True)
        
        # 格式化分數
        score_columns = ['final_score', 'sensor_type_similarity', 'module_similarity', 
                        'semantic_similarity', 'environment_similarity']
        for score_col in score_columns:
            if score_col in result.columns:
                result[score_col] = result[score_col].round(3)
        
        return result
        
    except Exception as e:
        print(f"推薦過程發生錯誤：{e}")
        return None

def calculate_module_similarity(user_input: str, df):
    similarities = []
    user_lower = user_input.lower()

    application_keywords = {
        '溫濕度監控偵測': {
            'primary': ['溫度', '濕度', '溫濕度', '環境溫度', '環境濕度'],
            'secondary': ['農業', '氣候', '環境監控'],
            'context': ['持續監測', '記錄', '感測'],
        },
        '火災預警偵測': {
            'primary': ['火災', '火源', '熱源', '安全監控', '預警', '熱顯像', 'co2', 'CO₂','二氧化碳','濃度'],
            'secondary': ['建築安全', '監控系統', '紅外線', '溫度監測'],
            'context': ['即時', '自動', '警報', '偵測'],
        },
        '火災預警偵測(wifi)': {
            'primary': ['火災', '無線', 'wifi', '遠端監控', '熱顯像', 'co2', 'CO₂' , '二氧化碳','濃度'],
            'secondary': ['物聯網', 'iot', '雲端', '無線傳輸'],
            'context': ['即時傳輸', '遠端', '無線通訊'],
        },
        '體溫偵測': {
            'primary': ['體溫', '紅外線感測', '熱顯像', '額溫偵測'],
            'secondary': ['發燒', '體表溫度', '醫療偵測', '健康監控'],
            'context': ['即時傳輸', '遠端', '不接觸'],
        },
        '氣候光照偵測': {
            'primary': ['光照', '照度', '環境光', '氣候', 'co2', 'CO₂','二氧化碳'],
            'secondary': ['農業', '植物生長', '溫室', '環境監測'],
            'context': ['光線感測', '氣候監控', '環境參數'],
        },
        '土壤氣候整合偵測': {
            'primary': ['土壤', '氣候', '農業', '種植', 'co2','CO₂', '光照', '溫濕度'],
            'secondary': ['智慧農業', '精準農業', '植物生長'],
            'context': ['整合監測', '多參數', '農業應用'],
        },
        '無CO2土壤氣候偵測': {
            'primary': ['土壤', '氣候', '光照', '環境監測', '無co2'],
            'secondary': ['簡化監測', '基礎農業', '環境感測'],
            'context': ['基本參數', '成本優化'],
        },
        '人流計算與氨氣感測': {
            'primary': ['人流', '人數統計', '氨氣', '毫米波', '雷達', '空氣品質', '空氣監控','空氣異味','有害氣體', '空氣','人潮','排風','換氣'],
            'secondary': ['空氣品質', '人員統計','通風','排風系統','除臭','空間使用'],
            'context': ['雷達偵測', '氣體監測', '雙重功能'],
        },
        '森林應用監測': {
            'primary': ['森林', '風力', '風向', '風速', '戶外監測'],
            'secondary': ['氣象', '環境監測', '野外應用'],
            'context': ['超音波', '氣象參數', '戶外環境'],
        },
        '傾斜偵測': {
            'primary': ['傾斜', '角度', '穩定性', '結構監測', '振動'],
            'secondary': ['建築安全', '結構健康', '設備監控'],
            'context': ['雙軸', '精密測量', '安全監控'],
        },
        '馬達偵測': {
            'primary': ['馬達', '振動', '設備監控', '機械', '傾斜'],
            'secondary': ['工業設備', '預測維護', '機械健康'],
            'context': ['設備診斷', '振動分析', '預防保養'],
        }
    }

    for _, row in df.iterrows():
        max_similarity = 0.0

        if 'parsed_modules' in row.index and row['parsed_modules']:
            for module in row['parsed_modules']:
                module_clean = re.sub(r'[（(].*?[）)]', '', module.lower())
                module_clean = module_clean.replace("偵測器", "偵測")

                if module_clean in user_lower:
                    max_similarity = max(max_similarity, 1.0)
                    continue

                matched_key = None
                for key in application_keywords:
                    if module_clean in key.lower() or key.lower() in module_clean:
                        matched_key = key
                        break

                if not matched_key:
                    matched_keys = get_close_matches(module_clean, [k.lower() for k in application_keywords.keys()], n=1, cutoff=0.6)
                    if matched_keys:
                        matched_key = next((k for k in application_keywords if k.lower() == matched_keys[0]), None)

                if matched_key:
                    keywords = application_keywords[matched_key]
                    primary = sum(1 for kw in keywords['primary'] if kw in user_lower)
                    secondary = sum(1 for kw in keywords['secondary'] if kw in user_lower)
                    context = sum(1 for kw in keywords['context'] if kw in user_lower)

                    weight = (primary * 1 + secondary * 0.2 + context * 0.05)
                    total = (len(keywords['primary']) * 0.4 + len(keywords['secondary']) * 0.07 + len(keywords['context']) * 0.02)
                    similarity = weight / total
                    max_similarity = max(max_similarity, similarity)

        similarities.append(min(max_similarity,1))

    return np.array(similarities)

def calculate_environment_similarity(user_input: str, df):
    similarities = []
    user_lower = user_input.lower()
    
    # 環境需求關鍵字
    env_requirements = {
        '低溫環境': ['低溫', '冷藏', '冷凍', '極低溫'],
        '高濕環境': ['高濕', '潮濕', '抗濕'],
        'AI擴充': ['AI', '人工智慧', '機器學習', '行為辨識'],
        '即時監控': ['即時', '實時', '連續監測'],
        '人員追蹤': ['人員', '移動軌跡', '追蹤', '行為']
    }
    
    for _, row in df.iterrows():
        env_score = 0.0
        
        # 低溫環境適用性
        if any(keyword in user_lower for keyword in env_requirements['低溫環境']):
            if pd.notna(row.get('operating_temp')):
                temp_range = str(row['operating_temp'])
                if '-' in temp_range:
                    try:
                        temps = re.findall(r'-?\d+', temp_range)
                        if len(temps) >= 2:
                            min_temp = int(temps[0])
                            if min_temp <= -20:  # 支援低溫
                                env_score += 0.3
                    except:
                        pass
        
        # 高濕環境抗性
        if any(keyword in user_lower for keyword in env_requirements['高濕環境']):
            if pd.notna(row.get('ip_rating')):
                ip_rating = str(row['ip_rating']).upper()
                if any(rating in ip_rating for rating in ['IP65', 'IP66', 'IPX7']):
                    env_score += 0.2
        
        # AI擴充能力
        if any(keyword in user_lower for keyword in env_requirements['AI擴充']):
            if pd.notna(row.get('features')):
                features = str(row['features']).lower()
                if any(ai_feature in features for ai_feature in ['分析', '智能', '擴充', '平台']):
                    env_score += 0.2
        
        # 即時監控能力
        if any(keyword in user_lower for keyword in env_requirements['即時監控']):
            if pd.notna(row.get('features')):
                features = str(row['features']).lower()
                if any(realtime_feature in features for realtime_feature in ['即時', '連續', '持續']):
                    env_score += 0.15
        
        similarities.append(min(env_score, 1.0))
    
    return np.array(similarities)


def interactive_recommend():
    print("=== 感測器智慧推薦系統 v2.0 ===")
    
    # 初始化系統
    df, model, device_embeddings = initialize_system()
    
    if df is None:
        print("系統初始化失敗，請檢查資料")
        return
    
    print(f"已載入 {len(df)} 款感測器\n")
    
    while True:
        try:
            user_input = input("🔍 請描述您的感測器應用需求：\n> ").strip()
            
            if user_input.lower() in ['q', 'quit', '離開']:
                print("感謝使用！")
                break
            
            if not user_input:
                continue
            
            # 意圖分析
            intent = analyze_user_intent(user_input)
            print(f"\n🧠 需求意圖分析：")
            if intent['direct_sensor_needs']:
                print(f"   感測器需求: {', '.join(intent['direct_sensor_needs'])}")
            #     weight_type=1
            # else:
            #     weight_type=2
            if intent['environmental_context']:
                print(f"   環境背景描述: {', '.join(intent['environmental_context'])}")
            detailed_analysis = analyze_user_intent(user_input)
            if detailed_analysis['primary_application']:
                print(f"   主要應用場景: {detailed_analysis['primary_application']}")
            
            print("\n🔄 正在分析並匹配感測器...")
            
            result = recommend_advanced(
                    user_input, df, model, device_embeddings,
                    sensor_type_weight=0.4,
                    module_weight=0.3,
                    semantic_weight=0.25,
                    environment_weight=0.05,
                    threshold=0.5,
                    top_k=5
                )
            
            
            if result is None or result.empty:
                print("❌ 很抱歉，沒有找到符合需求的感測器。")
                print("🔄 嘗試降低門檻重新搜尋...")
                
                relaxed_result = recommend_advanced(
                    user_input, df, model, device_embeddings, 
                    threshold=0.4, top_k=3
                )
                
                if relaxed_result is not None and len(relaxed_result) > 0:
                    print(f"找到 {len(relaxed_result)} 款可能相關的感測器：")
                    for idx, row in relaxed_result.iterrows():
                        print(f"  • {row['name']} (總評分: {row['final_score']:.3f})")
                        print(f"   詳細評分:")
                        print(f"      • 類型匹配: {row.get('sensor_type_similarity', 0):.3f}")
                        print(f"      • 模組相容: {row.get('module_similarity', 0):.3f}")
                        print(f"      • 環境適用: {row.get('environment_similarity', 0):.3f}")
                        print(f"      • 語意相似: {row.get('semantic_similarity', 0):.3f}")
                else:
                    print("❌ 很抱歉，沒有找到符合需求的感測器。")

                print("💡 建議：")
                print("   - 請嘗試使用更具體的關鍵字")
                print("   - 描述具體的應用場景")
                print("   - 說明環境條件需求")
                continue
            
            # 顯示推薦結果
            print(f"\n📊 找到 {len(result)} 款推薦感測器：")
            print("=" * 80)
            
            for idx, row in result.iterrows():
                print(f"\n【推薦 {idx+1} 】{row['name']}( {row['type']})")
                print(f"⭐ 綜合評分: {row['final_score']:.3f}")
                
                # 詳細評分分析
                print(f"📊 詳細評分:")
                print(f"   - 類型匹配度: {row['sensor_type_similarity']:.3f}")
                print(f"   - 模組匹配度: {row['module_similarity']:.3f}")
                print(f"   - 語意相似度: {row['semantic_similarity']:.3f}")
                print(f"   - 環境適用度: {row['environment_similarity']:.3f}")
                
                # 相容模組
                if row['parsed_modules']:
                    modules_str = ', '.join(row['parsed_modules'])
                    print(f"模組: {modules_str}")
                
                if pd.notna(row.get('features')):
                    print(f"✨ 主要特色: {row['features']}")
                
                if 'ip_rating' in row.index and pd.notna(row['ip_rating']):
                        ip_rating = row['ip_rating']
                        if ip_rating not in ['未標示', '未指定']:
                            print(f"   🛡️ 防護等級: {ip_rating}")
                            # IP等級說明
                            if 'IP65' in str(ip_rating):
                                print(f"      └─ 完全防塵，可防噴水 (適合室內外)")
                            elif 'IPX7' in str(ip_rating):
                                print(f"      └─ 可短時間浸水 (適合潮濕環境)")
                # 工作溫度
                if 'operating_temp' in row.index and pd.notna(row['operating_temp']):
                    print(f"   🌡️ 工作溫度: {row['operating_temp']}")
                    
                # 功耗
                if 'power_consumption' in row.index and pd.notna(row['power_consumption']):
                    power = row['power_consumption']
                    print(f"   🔋 功耗: {power}W", end="")
                    if float(power) <= 0.1:
                        print(" (低功耗，適合長期監控)")
                    else:
                        print()
                # 精度和範圍
                if 'range' in row.index and pd.notna(row['range']):
                    range_info = str(row['range'])[:100] + "..." if len(str(row['range'])) > 100 else str(row['range'])
                    print(f"   📏 量測範圍: {range_info}")
                    
                if 'precision' in row.index and pd.notna(row['precision']):
                    precision = str(row['precision'])[:100] + "..." if len(str(row['precision'])) > 100 else str(row['precision'])
                    print(f"   🎯 精度: {precision}")
                    
                 # 推薦說明
                if row['module_similarity'] > 0.5 or row['environment_similarity'] > 0.2:
                    print("-" * 60)
                    print(f"\n💡 推薦說明:")
                    row = result.iloc[0]
                    
                    if row['module_similarity'] > 0.5:
                        print(f"✅ 該感測器的相容模組適合您的應用場景")
                    
                    if row['environment_similarity'] > 0.2:
                        print(f"✅ 該感測器能適應您描述的環境條件")
                
            print("\n" + "=" * 80)
            
        except KeyboardInterrupt:
            print("\n\n程式已中斷")
            break
        except Exception as e:
            print(f"❌ 推薦過程發生錯誤：{e}")
            continue
class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    sensor_type_weight: float = Field(0.4, ge=0.0, le=1.0)
    module_weight: float = Field(0.3, ge=0.0, le=1.0)
    semantic_weight: float = Field(0.2, ge=0.0, le=1.0)
    environment_weight: float = Field(0.1, ge=0.0, le=1.0)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    top_k: int = Field(5, ge=1, le=20)


def main():
    """主程式入口"""
    try:
        interactive_recommend()
    except Exception as e:
        print(f"系統啟動失敗：{e}")

if __name__ == "__main__":
    main()