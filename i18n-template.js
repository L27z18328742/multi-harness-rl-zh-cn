// Localize only displayed text and accessible labels; source code and raw task files stay intact.
(() => {
  const dictionary = __DICTIONARY__;
  const fragments = Object.entries(dictionary).filter(([key]) => key.length > 7).sort((a,b) => b[0].length-a[0].length).map(([en,zh]) => {
    const escaped = en.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
    return [new RegExp((/^[A-Za-z]/.test(en)?'(?<![A-Za-z])':'') + escaped + (/[A-Za-z]$/.test(en)?'(?![A-Za-z])':''),'g'),zh];
  });
  const cache = new Map();
  function translate(value) {
    if (!/[A-Za-z]{2}/.test(value)) return value;
    if (cache.has(value)) return cache.get(value);
    const trim = value.trim();
    let result = dictionary[trim];
    if (result !== undefined) result = value.replace(trim, result);
    else {
      result = value;
      for (const [en, zh] of fragments) result = result.replace(en, () => zh);
      result = result
        .replace(/Turn (\d+) · model call/g, '第 $1 轮 · 模型调用')
        .replace(/Turn (\d+) · to vLLM/g, '第 $1 轮 · 发给 vLLM')
        .replace(/Turn (\d+) · recorded/g, '第 $1 轮 · 已记录')
        .replace(/Turn (\d+) · reply/g, '第 $1 轮 · 回复')
        .replace(/Turn (\d+) · tool call/g, '第 $1 轮 · 工具调用')
        .replace(/\bstep (\d+)/gi, '第 $1 步')
        .replace(/\bepoch (\d+)/gi, '第 $1 个 epoch')
        .replace(/(?<![\d.])(\d+(?:\.\d+)?) calls?\b(?! it)/g, '$1 次调用')
        .replace(/(\d+) steps?\b/g, '$1 步')
        .replace(/\btraining step\b/g, '训练步数')
        .replace(/\bcall(s)? per task\b/g, '每任务调用数')
        .replace(/\(a few cells ungraded\)/g, '（少数单元未评分）')
        .replace(/\brollout with (.*?) 次调用, (correct|wrong): reward (.*?), advantage (.*)/g, (_,n,ok,r,a) => `rollout：${n} 次调用，${ok==='correct'?'正确':'错误'}；奖励 ${r}，advantage ${a}`)
        .replace(/(\d+) of (\d+) recorded · text only/g,'记录 $1 / $2 个事件 · 仅文本')
        .replace(/(\d+) round trips/g,'$1 次往返')
        .replace(/(\d+) events in the episode/g,'过程中有 $1 个事件')
        .replace(/(\d+) 次调用 次调用的/g,'$1 次调用的')
        .replace(/\. r is 奖励\./g,'。r 表示奖励。')
        .replace(/task \(训练步数 (\d+)\)\./g,'任务（第 $1 个训练步骤）。')
        .replace(/^The trainer asks for one rollout: a task, a harness \((.*?)\) and a sandbox\.$/, '训练器请求一次 rollout：指定任务、harness（$1）和沙箱。')
        .replace(/^Harbor boots the sandbox, installs (.*?) and points it at the proxy with (.*?)\.$/, 'Harbor 启动沙箱，安装 $1，通过 $2 将其接到代理。')
        .replace(/^(.*?) calls what it thinks is a model API, in its own format \((.*?)\)\.$/, '$1 按自己的格式（$2）调用它认为的模型 API。')
        .replace(/^(.*?) calls the model again\. The prompt now ends with turn (\d+)’s tool result\.$/, '$1 再次调用模型；提示以第 $2 轮工具结果结尾。')
        .replace(/^The proxy records turn (\d+)\. Its prompt starts with turn (\d+)’s tokens exactly, so it is stored as turn (\d+)’s child\.$/, '代理记录第 $1 轮；提示以第 $2 轮的精确 token 开头，因此存为第 $3 轮的子节点。')
        .replace(/^The reply goes back to (.*?) in its own format, streamed if it asked for streaming\.$/, '按 $1 自己的格式返回回复；如请求流式，则以流式回放。')
        .replace(/^The reply goes back to (.*?) in its own format again\.$/, '再次按 $1 自己的格式返回回复。')
        .replace(/^The model asked to look at the data, so (.*?) runs a command in the sandbox\.$/, '模型请求查看数据，因此 $1 在沙箱中执行命令。')
        .replace(/^The trainer gets the reward and all (\d+) turns, each with its token ids, logprobs and loss mask\.$/, '训练器获得奖励和全部 $1 轮数据，每轮包含 token ID、logprobs 与损失掩码。');
    }
    if (cache.size < 4000) cache.set(value, result);
    return result;
  }
  function localize(root) {
    if (root.nodeType === Node.TEXT_NODE) {
      if (!root.parentElement?.closest('script,style,pre,code,textarea')) {
        const result = translate(root.nodeValue);
        if (result !== root.nodeValue) root.nodeValue = result;
      }
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE || root.matches('script,style,pre,code,textarea')) return;
    for (const attr of ['title','aria-label']) {
      const value = root.getAttribute(attr);
      if (value) { const result = translate(value); if (result !== value) root.setAttribute(attr,result); }
    }
    root.childNodes.forEach(localize);
  }
  const observer = new MutationObserver(records => {
    observer.disconnect();
    records.forEach(record => {
      if (record.type === 'childList') record.addedNodes.forEach(localize);
      else localize(record.target);
    });
    if (document.body) observer.observe(document.body, {subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['title','aria-label']});
  });
  document.addEventListener('DOMContentLoaded', () => {
    localize(document.body);
    observer.observe(document.body, {subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['title','aria-label']});
  });
})();
