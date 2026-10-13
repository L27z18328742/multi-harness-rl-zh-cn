(() => {
  let lastHeight = 0;
  function report() {
    const height = Math.ceil(document.body.getBoundingClientRect().height) + 8;
    if (height === lastHeight) return;
    lastHeight = height;
    parent.postMessage({type: 'embed-height', height}, location.origin);
  }
  new ResizeObserver(report).observe(document.body);
  window.addEventListener('load', report);
  report();
})();
