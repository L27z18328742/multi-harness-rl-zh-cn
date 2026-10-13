# 多 harness RL 终极指南 · 简体中文版

《[The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)》的非官方中文翻译。沿用本地 `ultrascale-playbook-zh-tw` 的暖白背景、紫色点缀、章节卡片、侧栏目录、分页阅读及随系统切换的深色主题。

原作者：Adithya S Kolavi、Joel Niklaus、Sergio Paniego Blanco、Leonie Monigatti、Amine Dirhoussi、Ben Burtenshaw、Lewis Tunstall、Leandro von Werra。机构：Hugging Face、Liquid AI。

原文发布于 2026 年 10 月 1 日，实验汇总截至 2026 年 9 月 29 日。本译本使用 2026 年 10 月 3 日下载的源码快照，revision 为 `f1d9a9311956925696736dde3d2721bec733d5b9`。文中版本状态、性能数据、实验局限均按此底本翻译，不代表对后续软件版本的承诺。

## 本地阅读

需要 Python 3 和 Node.js，无需 npm install。

```bash
cd /Users/lezewei/project/study/blog/multi-harness-rl-zh-cn
bash run.sh
```

打开 <http://127.0.0.1:8645>。更换端口：`PORT=8646 bash run.sh`。`Ctrl+C` 停止服务器。

也可以直接用浏览器打开 `webapp/index.html` 阅读；包含数据请求的图表需要通过上面的本地 HTTP 服务加载。

## 内容

- 六个正文部分：导论、基础知识、为什么跨 harness 训练、框架如何运作、跨 harness 训练小模型、结论。
- 保留全部 45 个章节标题、8 个代码块、36 个交互图表实例（25 种图表），包括 SFT 对比和 Qwen 早期实验的折叠内容。
- 分章页面、连续阅读全文、参考文献，以及可编辑的 Markdown 译稿。
- 本地保存原文图表实现、数据、模型与框架标志、论文截图和原文源码快照。

正文与图表界面采用简体中文。harness、rollout、token、logprob、GRPO 等术语适当保留英文，harness 首次出现时说明为“智能体运行框架”。代码与实际任务文件保持原样，论文截图和原始对比图片保留英文；这些是供核对的原始材料。文中 Harbor 在线任务浏览器仍指向原站，需要网络；正文、图表与图表数据在本地服务下无需访问外部 CDN。

## 目录

```text
content/                  六章 Markdown 译稿和图表中文词表
multi_harness_rl_zh.md     展开的中文全文，含原文代码、图表链接和参考文献
webapp/index.html         章节首页
webapp/full.html          连续阅读全文
webapp/chapters/          六章与参考文献 HTML
webapp/embeds/            36 个带对应配置的本地图表实例
webapp/assets/            阅读样式、图表数据、图片、标志与论文截图
src/                      固定 revision 的原始源码与来源记录
build.py                  从译稿和原文快照重建页面
i18n-template.js          图表界面文字本地化，保留程序和任务文件
verify.py                 检查本地链接、锚点、代码完整性和图表 JS 语法
run.sh                    构建并启动本地阅读服务器
qa/                       结构覆盖、验证结果及预览截图
```

译稿中的 `<Code n="…" />` 从固定底本提取原始代码；`<Embed n="…" />` 复用对应原文图表及其配置，标题和说明写在译稿中。修改译稿或图表词表后运行：

```bash
python3 build.py
python3 verify.py
```

## 署名与许可

原文采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，原始许可见 `LICENSE`，中文翻译采用相同内容许可。修改包括中文翻译、章节拆分、阅读布局与图表界面本地化。这是非官方译本，不表示原作者认可或背书。

阅读样式参考 [Twinkle AI Community 的 Ultra-Scale Playbook 中文版](https://github.com/ai-twinkle/ultrascale-playbook-zh-tw)，其 Apache 2.0 许可保存在 `webapp/assets/DESIGN-LICENSE`。Markdown 渲染器 marked 的 MIT 许可声明保留在文件头部。

第三方图像、品牌标志、论文截图不重新授权。原出处与说明保存在 `src/app/public/logos/README.md`、`src/app/public/papers/README.md`，对比图保留 Joel Niklaus 署名与原帖链接。
