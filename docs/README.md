# CANNBot-DSL 文档站

这是 `cannbot-dsl.gitcode.com` 的文档站源码。当前阶段包含站点框架、项目级内容和首批 API 文档，其余 API 将在核对实现后逐步补充。

## 本地开发

```bash
cd docs
npm install
npm run dev
```

开发服务器监听 `127.0.0.1:5173`。

## 构建与预览

```bash
npm run build
npm run preview
```

生产预览监听 `127.0.0.1:4174`，构建产物位于 `.vitepress/dist/`。

## 目录

```text
docs/
├── .vitepress/       # 站点配置与主题
├── public/           # 静态资源
├── getting-started/  # 开始使用
├── guide/            # 项目指南
├── examples/         # 样例导航
├── api/              # 公共 API 文档
├── community/        # 参与贡献
└── about/            # 版本与发布信息
```
