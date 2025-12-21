#!/usr/bin/env python3
"""
确定bias_detector模型的正确bias_class_index
使用对照文本测试哪个索引对应bias类
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def determine_bias_class_index(model_path="bias_detector_data"):
    """
    使用对照文本确定bias_class_index
    
    Args:
        model_path: 模型路径
        
    Returns:
        推荐的bias_class_index
    """
    print(f"🔍 Loading model from: {model_path}")
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    
    # 测试文本：中性 vs 偏见
    test_texts = [
        "This is a factual report about the economic situation.",  # 中性
        "Those corrupt politicians are destroying our country and should be stopped.",  # 偏见
        "The meeting was held at 3 PM yesterday.",  # 中性
        "These people are absolutely disgusting and dangerous to society.",  # 偏见
    ]
    
    print(f"📊 Model info:")
    print(f"   num_labels: {model.config.num_labels}")
    print(f"   id2label: {model.config.id2label}")
    print(f"   label2id: {model.config.label2id}")
    
    # 预测
    inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    
    print(f"\n📈 Prediction results:")
    labels = ["neutral", "biased", "neutral", "biased"]
    
    for i, (text, label) in enumerate(zip(test_texts, labels)):
        print(f"   {i+1}. [{label:7}] {text[:50]}...")
        print(f"      Probs: [0]={probs[i][0]:.3f}, [1]={probs[i][1]:.3f}")
    
    # 分析哪个索引更像bias类
    neutral_probs = probs[[0, 2]]  # 中性文本的概率
    biased_probs = probs[[1, 3]]   # 偏见文本的概率
    
    # 计算每个类别在偏见文本上的平均概率提升
    neutral_avg = neutral_probs.mean(axis=0)
    biased_avg = biased_probs.mean(axis=0)
    delta = biased_avg - neutral_avg
    
    print(f"\n📊 Analysis:")
    print(f"   Neutral texts avg: [0]={neutral_avg[0]:.3f}, [1]={neutral_avg[1]:.3f}")
    print(f"   Biased texts avg:  [0]={biased_avg[0]:.3f}, [1]={biased_avg[1]:.3f}")
    print(f"   Delta (biased-neutral): [0]={delta[0]:+.3f}, [1]={delta[1]:+.3f}")
    
    # 确定bias_class_index
    recommended_index = int(np.argmax(delta))
    confidence = abs(delta[recommended_index])
    
    print(f"\n💡 Recommendation:")
    print(f"   bias_class_index = {recommended_index}")
    print(f"   Confidence: {confidence:.3f}")
    
    if confidence < 0.1:
        print(f"   ⚠️  Low confidence - you may want to test with more specific texts")
    else:
        print(f"   ✅ High confidence - index {recommended_index} clearly corresponds to bias class")
    
    # 生成配置代码
    print(f"\n🔧 Configuration code:")
    print(f"   # Add this to your config:")
    print(f"   config.teacher.bias_class_index = {recommended_index}")
    
    return recommended_index

if __name__ == "__main__":
    try:
        recommended_index = determine_bias_class_index()
        print(f"\n✅ Recommended bias_class_index: {recommended_index}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the model is properly loaded and accessible.")