"""
SV2000 Framing Head - 基于Semetko & Valkenburg (2000)的框架预测组件
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class SVFramingHead:
    """SV2000框架预测头 - 主要的框架分析组件"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.encoder_backend = None  # 用于日志展示当前编码器类型
        self.temperature = torch.ones(5, device=self.device)
        self.logit_bias = torch.zeros(5, device=self.device)
        self.enable_encoder_grad = False
        self.uses_item_level = getattr(self.config, "training_mode", "frame_level") == "item_level"
        self.frame_count = 5
        self.item_count = 20
        self.frame_to_items = {
            "conflict": [1, 2, 3, 4],
            "human": [5, 6, 7, 8, 9],
            "econ": [10, 11, 12],
            "moral": [13, 14, 15],
            "resp": [16, 17, 18, 19, 20],
        }
        self.frame_agg_matrix = self._build_aggregation_matrix()
        
        # 加载编码器和分类器
        self.encoder = None
        self.frame_classifier = None
        self._load_components()
        
        logger.info("SVFramingHead initialized successfully")
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA不可用，自动回退CPU")
                device = "cpu"
        
        logger.info(f"SVFramingHead using device: {device}")
        return device
    
    def _load_components(self):
        """加载编码器和分类器组件"""
        try:
            encoder_name_lower = str(self.config.encoder_name).lower()
            is_sentence_encoder = any(
                marker in encoder_name_lower for marker in ["sentence-transformers", "all-minilm", "bge"]
            )
            use_sentence_transformers = (
                SentenceTransformer is not None
                and not getattr(self.config, "fine_tune_encoder", False)
                and is_sentence_encoder
            )

            # 加载编码器（优先使用本地路径）
            encoder_path = self.config.encoder_local_path or self.config.encoder_name
            logger.info(f"Loading encoder from: {encoder_path}")

            # 校验本地目录是否为合法的sentence-transformers结构，否则回退transformers后端
            if (
                use_sentence_transformers
                and Path(encoder_path).exists()
                and not self._is_valid_sentence_transformer_dir(encoder_path)
            ):
                logger.warning(
                    "检测到本地编码器目录缺少sentence-transformers模块，已回退为transformers后端: %s",
                    encoder_path,
                )
                use_sentence_transformers = False

            if use_sentence_transformers:
                try:
                    # 使用sentence-transformers
                    self.encoder_backend = "sentence_transformers"
                    self.encoder = SentenceTransformer(encoder_path)
                    self.encoder.to(self.device)
                    # 限制最大序列长度，避免长文本超出模型位置嵌入
                    requested_max_length = getattr(self.config, "max_length", 512)
                    model_max_length = self._infer_model_max_length(self.encoder)
                    effective_max_length = min(requested_max_length, model_max_length)
                    if requested_max_length > model_max_length:
                        logger.warning(
                            "Requested max_length %s exceeds encoder limit %s; using %s instead",
                            requested_max_length,
                            model_max_length,
                            effective_max_length,
                        )
                    self.encoder.max_seq_length = effective_max_length
                    # 暴露tokenizer供长度统计使用，并同步最大长度
                    self.tokenizer = getattr(self.encoder, "tokenizer", None)
                    if self.tokenizer is not None and hasattr(self.tokenizer, "model_max_length"):
                        self.tokenizer.model_max_length = effective_max_length
                    hidden_size = self.encoder.get_sentence_embedding_dimension()
                except Exception as load_err:
                    logger.warning(
                        "sentence-transformers加载失败，已回退transformers后端: %s",
                        load_err,
                    )
                    use_sentence_transformers = False

            if not use_sentence_transformers:
                # 使用标准transformers
                self.encoder_backend = "transformers"
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
                    self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
                except Exception as load_err:
                    fallback_path = self._get_local_fallback_model_path(encoder_path)
                    if fallback_path:
                        logger.warning(
                            "加载编码器 %s 失败，将回退至本地默认模型 %s：%s",
                            encoder_path,
                            fallback_path,
                            load_err,
                        )
                        self.config.encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
                        self.config.encoder_local_path = fallback_path
                        self.tokenizer = AutoTokenizer.from_pretrained(fallback_path, local_files_only=True)
                        self.encoder = AutoModel.from_pretrained(fallback_path, local_files_only=True)
                    else:
                        raise
                self.encoder.to(self.device)
                requested_max_length = getattr(self.config, "max_length", 512)
                model_max_length = self._infer_model_max_length(self.encoder)
                effective_max_length = min(requested_max_length, model_max_length)
                if requested_max_length > model_max_length:
                    logger.warning(
                        "Requested max_length %s exceeds encoder limit %s; using %s instead",
                        requested_max_length,
                        model_max_length,
                        effective_max_length,
                    )
                if hasattr(self.tokenizer, "model_max_length"):
                    self.tokenizer.model_max_length = effective_max_length
                hidden_size = self.encoder.config.hidden_size

            # 如果需要微调且使用sentence-transformers，提醒限制
            if getattr(self.config, "fine_tune_encoder", False) and self.encoder_backend == "sentence_transformers":
                logger.warning("fine_tune_encoder 启用但当前编码器为 sentence-transformers，encode 接口不支持反向传播；已仅训练分类头部")
                self.config.fine_tune_encoder = False
                self.config.trainable_encoder_layers = 0

            # 控制编码器可训练性
            self.enable_encoder_grad = self._set_encoder_trainable(
                trainable=getattr(self.config, "fine_tune_encoder", False) 
                or getattr(self.config, "trainable_encoder_layers", 0) > 0,
                trainable_layers=getattr(self.config, "trainable_encoder_layers", 0)
            )
            
            # 创建框架分类器
            output_dim = self.item_count if self.uses_item_level else self.frame_count
            self.frame_classifier = nn.Sequential(
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(hidden_size, output_dim)
            ).to(self.device)
            
            # 初始化分类器权重
            nn.init.xavier_uniform_(self.frame_classifier[1].weight)
            nn.init.zeros_(self.frame_classifier[1].bias)
            
            logger.info(f"Frame classifier initialized with hidden_size={hidden_size}, output_dim={output_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load SV2000 components: {e}")
            raise

    def _build_aggregation_matrix(self) -> torch.Tensor:
        """构建题项到框架的平均池化矩阵"""
        mat = torch.zeros(self.frame_count, self.item_count, dtype=torch.float32)
        for frame_idx, frame_name in enumerate(["conflict", "human", "econ", "moral", "resp"]):
            items = self.frame_to_items.get(frame_name, [])
            if not items:
                continue
            weight = 1.0 / len(items)
            for item_id in items:
                mat[frame_idx, item_id - 1] = weight
        return mat

    def _infer_model_max_length(self, encoder) -> int:
        """推断编码器允许的最大长度，避免超出位置嵌入"""
        candidates = []

        tokenizer = getattr(encoder, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "model_max_length"):
            candidates.append(tokenizer.model_max_length)

        auto_model = getattr(encoder, "auto_model", None)
        if auto_model is not None and hasattr(auto_model, "config"):
            max_pos = getattr(auto_model.config, "max_position_embeddings", None)
            if max_pos is not None:
                candidates.append(max_pos)

        # SentenceTransformer没有auto_model时，尝试直接读取config
        model_config = getattr(encoder, "config", None)
        if model_config is not None:
            max_pos = getattr(model_config, "max_position_embeddings", None)
            if max_pos is not None:
                candidates.append(max_pos)

        valid_candidates = [c for c in candidates if isinstance(c, int) and c > 0 and c < 1_000_000]
        if not valid_candidates:
            return getattr(self.config, "max_length", 512)

        return min(valid_candidates)

    def _is_valid_sentence_transformer_dir(self, path: str) -> bool:
        """检测本地目录是否包含完整的sentence-transformers模块结构"""
        path_obj = Path(path)
        modules_file = path_obj / "modules.json"
        if not modules_file.exists():
            return False

        try:
            modules = json.loads(modules_file.read_text())
        except Exception as exc:
            logger.warning("无法读取modules.json (%s): %s", modules_file, exc)
            return False

        for module in modules:
            module_path = module.get("path", "")
            if not module_path:
                # 主Transformer模块应当存在配置文件
                if not (path_obj / "config.json").exists():
                    return False
                continue
            if not (path_obj / module_path).exists():
                logger.warning("缺少sentence-transformers模块目录: %s", path_obj / module_path)
                return False

        return True

    def _get_local_fallback_model_path(self, current_path: str) -> Optional[str]:
        """
        在离线环境下为编码器加载提供可靠的本地回退路径。
        优先使用仓库自带的 all-MiniLM-L6-v2。
        """
        fallback_dir = Path(__file__).resolve().parent / "all-MiniLM-L6-v2"
        if fallback_dir.exists() and str(fallback_dir) != str(current_path):
            return str(fallback_dir)
        return None

    def _set_encoder_trainable(self, trainable: bool, trainable_layers: int = 0) -> bool:
        """控制编码器参数是否参与训练，支持仅解冻末尾若干层"""
        if self.encoder is None or not hasattr(self.encoder, "parameters"):
            return False

        # 对sentence-transformers后台，保持冻结（encode不支持反向传播）
        if self.encoder_backend != "transformers" and (trainable or trainable_layers > 0):
            logger.warning("编码器后端 %s 不支持梯度微调，已保持冻结", self.encoder_backend)
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.config.trainable_encoder_layers = 0
            return False

        if not trainable and trainable_layers <= 0:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder parameters set to frozen")
            return False

        if trainable_layers <= 0:
            for param in self.encoder.parameters():
                param.requires_grad = True
            logger.info("Encoder parameters set to trainable (all layers)")
            return True

        # 部分解冻：先全部冻结，再解冻末尾若干层
        for param in self.encoder.parameters():
            param.requires_grad = False

        layer_container = None
        for attr in ("encoder", "transformer"):
            sub = getattr(self.encoder, attr, None)
            if sub is not None and hasattr(sub, "layer"):
                layer_container = sub.layer
                break

        if layer_container is not None:
            layers = list(layer_container)[-trainable_layers:]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = True
            if hasattr(self.encoder, "pooler"):
                for param in self.encoder.pooler.parameters():
                    param.requires_grad = True
            logger.info("Encoder parameters set to partially trainable (last %d layers)", len(layers))
            return True

        # 回退：若未找到layer容器则全部解冻，避免完全无法训练
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.warning("未找到 encoder.layer/transformer.layer，已全部解冻作为回退")
        return True
    
    def predict_frames(self, texts: List[str], return_logits: bool = False):
        """预测SV2000框架分数
        
        Args:
            texts: 输入文本列表
            return_logits: 是否同时返回原始logits（用于评估/损失计算）
            
        Returns:
            - 默认返回包含5个框架分数和平均分数的字典
            - 当return_logits为True时，返回 (预测字典, raw_logits, frame_logits)
        """
        if not texts:
            if return_logits:
                return self._empty_predictions(0), None, None
            return self._empty_predictions(0)
        
        try:
            # 获取文本嵌入
            embeddings = self._encode_texts(texts)
            
            # 框架分类预测
            with torch.inference_mode():
                raw_logits = self.frame_classifier(embeddings)
                frame_logits = self.aggregate_item_logits(raw_logits) if self.uses_item_level else raw_logits
                calibrated_logits = self.apply_calibration(frame_logits)
                frame_probs = torch.sigmoid(calibrated_logits)
                
                # 转换为numpy数组
                frame_probs_np = frame_probs.cpu().numpy()
                
                # 计算平均框架分数
                frame_avg = np.mean(frame_probs_np, axis=1)
                outputs = {
                    'sv_conflict_pred': frame_probs_np[:, 0],
                    'sv_human_pred': frame_probs_np[:, 1],
                    'sv_econ_pred': frame_probs_np[:, 2],
                    'sv_moral_pred': frame_probs_np[:, 3],
                    'sv_resp_pred': frame_probs_np[:, 4],
                    'sv_frame_avg_pred': frame_avg
                }
                if self.uses_item_level and getattr(self.config, "return_item_predictions", False):
                    item_probs = torch.sigmoid(raw_logits).cpu().numpy()
                    outputs["sv_item_pred"] = item_probs
                if return_logits:
                    return outputs, raw_logits.detach(), frame_logits.detach()
                return outputs
                
        except Exception as e:
            logger.error(f"Error in frame prediction: {e}")
            if return_logits:
                return self._empty_predictions(len(texts)), None, None
            return self._empty_predictions(len(texts))
    
    def _encode_texts(self, texts: List[str], requires_grad: bool = False) -> torch.Tensor:
        """编码文本为嵌入向量，长文本自动分片后平均池化"""
        if self.encoder_backend == "sentence_transformers" and hasattr(self.encoder, "encode"):
            return self._encode_with_sentence_transformer(texts)
        return self._encode_with_transformer(texts, requires_grad=requires_grad)

    def _tokenize_without_warning(self, text: str) -> List[int]:
        """在分片前安全获取tokens，避免超长警告"""
        if self.tokenizer is None:
            return []
        original_max_length = getattr(self.tokenizer, "model_max_length", None)
        try:
            # 临时放宽长度限制，仅用于离线分片，不影响推理阶段的真实截断
            if original_max_length is not None and original_max_length < 1_000_000:
                self.tokenizer.model_max_length = 1_000_000
            return self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
                max_length=None
            )
        finally:
            if original_max_length is not None:
                self.tokenizer.model_max_length = original_max_length

    def _encode_with_sentence_transformer(self, texts: List[str]) -> torch.Tensor:
        """使用sentence-transformers编码并对长文本分片平均"""
        pooled_embeddings = []
        batch_size = self.config.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            expanded_texts = []
            chunk_counts = []
            for text in batch:
                chunks = self._chunk_text_by_tokens(text)
                expanded_texts.extend(chunks)
                chunk_counts.append(len(chunks))

            chunk_embeddings = self.encoder.encode(
                expanded_texts,
                convert_to_tensor=True,
                device=self.device,
                batch_size=batch_size
            )

            offset = 0
            for count in chunk_counts:
                slice_embeddings = chunk_embeddings[offset:offset + count]
                pooled = slice_embeddings.mean(dim=0) if count > 1 else slice_embeddings[0]
                pooled_embeddings.append(pooled)
                offset += count

        return torch.stack(pooled_embeddings, dim=0)

    def _encode_with_transformer(self, texts: List[str], requires_grad: bool = False) -> torch.Tensor:
        """使用标准transformers编码，保留反向传播能力"""
        embeddings_list = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.set_grad_enabled(requires_grad):
                outputs = self.encoder(**inputs)
                # 使用平均池化提升稳定性
                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                masked_sum = (last_hidden * mask).sum(dim=1)
                mask_length = mask.sum(dim=1).clamp(min=1)
                batch_embeddings = masked_sum / mask_length
                embeddings_list.append(batch_embeddings)

        return torch.cat(embeddings_list, dim=0)

    def _chunk_text_by_tokens(self, text: str) -> List[str]:
        """按token长度对长文本分片，减少截断信息损失"""
        if self.tokenizer is None:
            return [text]

        tokens = self._tokenize_without_warning(text)
        max_len = max(8, getattr(self.config, "max_length", 512) - 2)
        if len(tokens) <= max_len:
            return [text]

        stride = getattr(self.config, "chunk_stride", max_len // 2)
        stride = max(1, min(stride, max_len - 1))
        max_chunks = max(1, getattr(self.config, "max_chunks_per_text", 4))
        step = max_len - stride

        chunks = []
        start = 0
        for _ in range(max_chunks):
            if start >= len(tokens):
                break
            sub_tokens = tokens[start:start + max_len]
            chunks.append(self.tokenizer.decode(sub_tokens, skip_special_tokens=True))
            if start + max_len >= len(tokens):
                break
            start += step

        return chunks or [text]

    def aggregate_item_logits(self, item_logits: torch.Tensor) -> torch.Tensor:
        """将20题项logits聚合为5个框架logits"""
        if not self.uses_item_level:
            return item_logits
        agg = self.frame_agg_matrix.to(item_logits.device)
        return torch.matmul(item_logits, agg.t())

    def aggregate_item_targets(self, item_targets: torch.Tensor) -> torch.Tensor:
        """将题项标签聚合为框架标签"""
        if not self.uses_item_level:
            return item_targets
        agg = self.frame_agg_matrix.to(item_targets.device)
        return torch.matmul(item_targets, agg.t())
    
    def _empty_predictions(self, batch_size: int) -> Dict[str, np.ndarray]:
        """返回空预测结果"""
        preds = {
            'sv_conflict_pred': np.zeros(batch_size),
            'sv_human_pred': np.zeros(batch_size),
            'sv_econ_pred': np.zeros(batch_size),
            'sv_moral_pred': np.zeros(batch_size),
            'sv_resp_pred': np.zeros(batch_size),
            'sv_frame_avg_pred': np.zeros(batch_size)
        }
        if self.uses_item_level and getattr(self.config, "return_item_predictions", False):
            preds['sv_item_pred'] = np.zeros((batch_size, self.item_count))
        return preds

    def set_temperature(self, temperature: torch.Tensor):
        """设置温度标定参数"""
        if temperature is None:
            return
        if temperature.shape[-1] != 5:
            raise ValueError("temperature 向量长度必须为5")
        self.temperature = temperature.to(self.device)
        logger.info("Applied temperature scaling: %s", self.temperature.detach().cpu().tolist())
    
    def set_logit_bias(self, bias: torch.Tensor):
        """设置logit偏置，配合温度缩放校准均值"""
        if bias is None:
            return
        if bias.shape[-1] != 5:
            raise ValueError("bias 向量长度必须为5")
        self.logit_bias = bias.to(self.device)
        logger.info("Applied logit bias: %s", self.logit_bias.detach().cpu().tolist())

    def apply_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        """统一应用温度缩放与偏置，避免多处重复逻辑"""
        calibrated = logits
        if self.temperature is not None:
            calibrated = calibrated / self.temperature.clamp(min=1e-3)
        if self.logit_bias is not None:
            calibrated = calibrated + self.logit_bias
        return calibrated
    
    def train_step(self, texts: List[str], targets: np.ndarray, optimizer, criterion) -> float:
        """单步训练
        
        Args:
            texts: 输入文本
            targets: 目标标签 (batch_size, 5)
            optimizer: 优化器
            criterion: 损失函数
            
        Returns:
            训练损失
        """
        self.frame_classifier.train()
        if self.enable_encoder_grad and self.encoder is not None:
            self.encoder.train()
        
        # 获取嵌入
        embeddings = self._encode_texts(texts, requires_grad=self.enable_encoder_grad)
        
        # 前向传播
        logits = self.frame_classifier(embeddings)
        
        # 计算损失
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        loss = criterion(logits, targets_tensor)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        max_grad_norm = getattr(self.config, "max_grad_norm", None)
        if max_grad_norm is not None and max_grad_norm > 0:
            params_to_clip = list(self.frame_classifier.parameters())
            if self.enable_encoder_grad and self.encoder is not None:
                params_to_clip += list(self.encoder.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
        optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'frame_classifier_state_dict': self.frame_classifier.state_dict(),
            'config': self.config,
            'temperature': self.temperature.detach().cpu(),
            'logit_bias': self.logit_bias.detach().cpu()
        }, path)
        logger.info(f"SV2000 model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.frame_classifier.load_state_dict(checkpoint['frame_classifier_state_dict'])
        # 若存在温度标定参数则加载
        if 'temperature' in checkpoint:
            self.set_temperature(checkpoint['temperature'])
        # logit偏置为可选，兼容旧模型
        if 'logit_bias' in checkpoint:
            self.set_logit_bias(checkpoint['logit_bias'])
        logger.info(f"SV2000 model loaded from {path}")
    
    def eval_mode(self):
        """设置为评估模式"""
        if self.frame_classifier:
            self.frame_classifier.eval()
        if self.encoder is not None:
            self.encoder.eval()
    
    def train_mode(self):
        """设置为训练模式"""
        if self.frame_classifier:
            self.frame_classifier.train()
        if getattr(self.config, "fine_tune_encoder", False) and self.encoder is not None:
            self.encoder.train()
