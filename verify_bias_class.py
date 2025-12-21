#!/usr/bin/env python3
"""
验证bias_detector模型的bias类别索引
使用对照句确定哪个索引对应bias类别
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def verify_bias_class_index(model_path="bias_detector_data"):
    """验证bias类别索引"""
    
    print("🔍 正在验证bias类别索引...")
    print(f"📁 模型路径: {model_path}")
    
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用设备: {device}")
    
    # 加载模型
    print("📥 加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        model = model.to(device).eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 测试文本
    test_texts = [
        "This is a factual report about the event.",  # 偏中性
        "Those people are disgusting and should be punished.",  # 明显偏见/攻击性
    ]
    
    print("\n📝 测试文本:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")
    
    # 推理
    print("\n🧠 进行推理...")
    inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.inference_mode():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    
    # 分析结果
    print("\n📊 分析结果:")
    print(f"模型配置:")
    print(f"  num_labels: {model.config.num_labels}")
    print(f"  id2label: {getattr(model.config, 'id2label', 'N/A')}")
    print(f"  label2id: {getattr(model.config, 'label2id', 'N/A')}")
    
    print(f"\n概率分布:")
    print(f"  中性文本: {probs[0]}")
    print(f"  偏见文本: {probs[1]}")
    
    # 计算差值
    delta = probs[1] - probs[0]
    print(f"\n差值 (偏见-中性): {delta}")
    
    # 推荐配置
    print("\n💡 推荐配置:")
    
    if model.config.num_labels == 2:
        # 找出在偏见文本中概率提升最大的索引
        max_increase_idx = np.argmax(delta)
        max_increase = delta[max_increase_idx]
        
        if max_increase > 0.1:  # 显著差异
            print(f"✅ 推荐使用 bias_class_index = {max_increase_idx}")
            print(f"   理由: 在偏见文本中，索引{max_increase_idx}的概率提升了 {max_increase:.3f}")
            
            print(f"\n🔧 配置方法:")
            print(f"   方法1 - 在代码中设置:")
            print(f"   config = AnalyzerConfig()")
            print(f"   config.teacher.bias_class_index = {max_increase_idx}")
            
            print(f"\n   方法2 - 在JSON配置文件中设置:")
            print(f"   \"teacher\": {{")
            print(f"     \"bias_class_index\": {max_increase_idx}")
            print(f"   }}")
            
            return {
                'recommended_index': int(max_increase_idx),
                'confidence': float(max_increase),
                'probabilities': probs.tolist(),
                'delta': delta.tolist()
            }
        else:
            print("⚠️  两个类别的差异不够明显，可能需要更明确的测试文本")
            print(f"   最大差异: {max_increase:.3f} (建议 > 0.1)")
            return None
    else:
        print(f"⚠️  模型有 {model.config.num_labels} 个类别，需要手动分析")
        return None

if __name__ == "__main__":
    result = verify_bias_class_index()
    
    if result:
        print(f"\n🎉 验证完成！推荐使用 bias_class_index = {result['recommended_index']}")
    else:
        print(f"\n❌ 无法确定推荐配置，请手动分析结果")