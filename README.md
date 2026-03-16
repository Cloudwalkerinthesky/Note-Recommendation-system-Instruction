# 仿小红书笔记推荐系统
本仓库为真实笔记推荐系统的项目说明与架构展示。  
由于毕业设计相关要求，核心代码仓库目前设置为 **Private**。

如需查看系统运行效果，可访问在线 Demo：

http://47.238.69.176

如需代码访问权限，可在面试或交流时提供。

本项目是一个毕业设计项目，旨在构建一个类似于小红书的个性化笔记推荐系统。系统使用来自 Amazon Beauty 2014 的真实数据，结合工业级深度学习架构，实现从召回到精排的完整推荐链路。

目前已升级至 **v0.5** 阶段，核心亮点为引入 **多领域海量数据集（美妆+服饰+健康）**，在近 200 万物品池和 20 万活跃用户规模下，成功训练并部署了 **Two-Tower（双塔）召回模型** 和 **Wide & Deep 精排模型**，形成完整的工业级两阶段推荐系统。

## 功能特性 (v0.5 Update)

*   **海量多领域数据支持**：融合 Amazon Beauty, Clothing, Health 三大类目，支持 21.8 万活跃用户、202 万海量笔记和 195 万次真实交互。
*   **工业级两阶段推荐架构**：
    *   **召回阶段**：双塔模型（DSSM）+ 热门召回，从 202 万物品中毫秒级筛选出 ~1000 个候选。在全量商品池中 Recall@500 达到 4.72%（随机召回的 191 倍）。
    *   **精排阶段**：Wide & Deep 模型对候选集逐一打分，选出 Top-K。HR@10 达到 93.45%，NDCG@10 达到 0.6405。
*   **Two-Tower 双塔召回模型**：User Tower 和 Item Tower 分别学习用户偏好向量和物品语义向量，通过 Faiss ANN 检索实现个性化召回（相比原始 BERT 加权平均，召回准确率大幅提升，Val Acc 从 50% 提升到 85.9%）
*   **Wide & Deep 精排模型**：Wide 侧（ID Embedding）+ Deep 侧（BERT 768 维文本向量经 MLP 压缩），联合学习用户-物品交互
*   **BERT 语义特征**：使用预训练 DistilBERT 提取笔记文本的 768 维语义向量，离线提取并作为物品特征
*   **Faiss ANN 向量检索**：IVFFlat 索引（内积距离），支持毫秒级近似最近邻检索
*   **历史过滤**：推荐时自动屏蔽用户已交互过的物品
*   **冷启动处理**：新用户返回热门推荐，保证服务可用性
*   **工业级数据处理**：5-core 过滤、时序 80/20 划分、1:4 负采样、Early Stopping

## 技术栈

*   **算法后端**：Python, FastAPI, Uvicorn
*   **深度学习**：PyTorch（Two-Tower 双塔模型、Wide & Deep 模型）
*   **NLP**：Hugging Face Transformers（DistilBERT 特征提取）
*   **向量检索**：Faiss（IVFFlat 近似最近邻索引）
*   **Java 后端**：Spring Boot, JPA, Redis, JWT
*   **前端**：Vue.js, Vite
*   **数据库**：MySQL, Redis
*   **数据处理**：Pandas, NumPy, Scikit-learn
*   **部署**：Nginx, Systemd, Ubuntu 22.04（阿里云 ECS）

## 项目结构

```
.
├── data/                         # 数据存储（大文件已 .gitignore）
│   ├── raw/                      # 原始数据集 (meta_Beauty.json, reviews_Beauty.json)
│   ├── interactions.csv          # 清洗后的用户-物品交互数据
│   ├── notes.csv                 # 笔记元数据
│   ├── users.csv                 # 用户元数据
│   ├── user_sequences.csv        # 用户行为序列
│   ├── note_embeddings.npy       # BERT 提取的 768 维文本向量（需自行生成）
│   ├── item_tower_embeddings.npy # Two-Tower 学习到的 64 维物品向量（需自行生成）
│   ├── faiss_tower_index.bin     # 双塔 Faiss 索引（需自行生成）
│   ├── faiss_tower_id_map.json   # Faiss 行号 → note_id 映射
│   ├── two_tower_model.pth       # 训练好的双塔模型权重（需自行生成）
│   └── wide_deep_model.pth       # 训练好的 Wide&Deep 模型权重（需自行生成）
├── src/
│   ├── api/
│   │   └── main.py               # FastAPI 算法服务（召回+精排完整链路）
│   ├── models/
│   │   ├── two_tower.py          # Two-Tower 双塔模型定义
│   │   └── wide_and_deep.py      # Wide & Deep 精排模型定义
│   └── scripts/
│       ├── process_amazon_data.py    # 数据清洗 ETL
│       ├── extract_features.py       # BERT 特征提取（支持断点续传）
│       ├── train_two_tower.py        # 双塔模型训练
│       ├── build_tower_faiss_index.py # 用双塔向量重建 Faiss 索引
│       ├── train_wide_deep.py        # Wide & Deep 模型训练
│       ├── build_faiss_index.py      # 用 BERT 向量构建 Faiss 索引（旧版，备用）
│       └── evaluate_model.py         # 离线评估（HR@10, NDCG@10）
├── Cooperation_Backend_Frontend/ # 合作者 Java Spring Boot + Vue 前后端
├── requirements.txt
├── README.md
└── DEVELOPMENT_LOG.md
```

## 快速开始 (v0.5 海量多领域数据版)

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 数据准备

前往 [Amazon Product Data (Julian McAuley)](http://jmcauley.ucsd.edu/data/amazon/) 下载 2014 版 Beauty, Clothing, Health 数据集：
- 将对应的 `reviews_*.json.gz` 和 `meta_*.json.gz` 解压到 `data/raw/` 目录下。

### 3. 数据清洗与特征工程

```bash
# 步骤 1：多数据集融合与清洗，生成 users/notes/interactions.csv
python src/scripts/process_multi_category.py

# 步骤 2：BERT 特征提取（支持断点续传，200万数据耗时约1小时）
python src/scripts/extract_features.py
```

### 4. 模型训练（两阶段）

```bash
# 阶段 A：训练 Wide & Deep 精排模型
python src/scripts/train_wide_deep.py

# 阶段 B：训练 Two-Tower 双塔召回模型，并生成全量物品向量
python src/scripts/train_two_tower.py

# 阶段 C：用双塔物品向量重建 Faiss 索引
python src/scripts/build_tower_faiss_index.py
```

### 5. 启动 Python 算法服务

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/health` 确认服务状态，应看到 `"recall_mode": "two_tower"`。

### 6. 启动完整系统（含 Java 后端 + Vue 前端）

```bash
# Java 后端（需先配置 MySQL，确保数据已导入）
cd Cooperation_Backend_Frontend/Final_year_project
export MYSQL_PASSWORD=your_password
mvn spring-boot:run

# Vue 前端
cd Cooperation_Backend_Frontend/Final_year_project/UI
npm install && npm run dev
```

## 离线评估结果 (21.8万用户, 202万物品)

### 召回层 (Two-Tower)
| 指标 | 结果 | 相对随机召回倍数 |
|------|------|----------------|
| Val Acc | 92.41% | - |
| Recall@100 | 1.30% | 263x |
| Recall@500 | 4.72% | 191x |

### 排序层 (Wide & Deep, 100候选集)
| 模型 | HR@10 | NDCG@10 |
|------|-------|---------|
| Random | 0.1007 | 0.0456 |
| Popularity | 0.9976 | 0.7306 |
| Wide & Deep | **0.9345** | **0.6405** |

### 7. 线上部署信息

本项目目前已在阿里云 ECS（Ubuntu 22.04, 4vCPU 16GB）上完成了全栈手工部署。系统通过 Git 与软连接（Symlink）管理代码目录，底层依赖 MySQL 8.0（通过环境变量注入密码）与 Redis 提供数据与缓存支持；核心算法层使用 FastAPI 挂载于 8000 端口，在内存中加载 200 万规模的 Faiss 索引与模型权重提供毫秒级检索；业务后端基于 Spring Boot 运行于 8080 端口处理鉴权与 API 聚合；前端 Vue.js 经 Node 20 编译后由 Nginx 统一托管静态资源，并通过 proxy_pass 解决跨域反向代理。为解决当前 nohup 进程管理脆弱与环境隔离差的问题，下一步计划引入 Docker 与 K3s 进行云原生微服务重构。



## 许可证

本项目仅用于教育目的。
