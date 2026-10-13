(() => {
  let anchored = Boolean(location.hash);
  ['wheel','touchstart','pointerdown','keydown'].forEach(type => window.addEventListener(type, () => {anchored = false;}, {passive:true}));
  // Frames report only their own size; accept messages from known same-origin frames.
  window.addEventListener('message', (event) => {
    if (event.origin !== location.origin || event.data?.type !== 'embed-height') return;
    const frame = [...document.querySelectorAll('.viz-frame')].find(f => f.contentWindow === event.source);
    if (frame && Number.isFinite(event.data.height)) {
      frame.style.height = Math.min(6000, Math.max(100, event.data.height)) + 'px';
      if (anchored) reveal();
    }
  });
  const headings = [...document.querySelectorAll('.prose h2[id],.prose h3[id],.prose h4[id]')];
  const links = [...document.querySelectorAll('.toc a')];
  const spy = new IntersectionObserver(entries => {
    const entry = entries.find(e => e.isIntersecting);
    if (entry) links.forEach(a => a.classList.toggle('active', a.hash === '#' + entry.target.id));
  }, {rootMargin: '-10% 0px -75% 0px'});
  headings.forEach(h => spy.observe(h));
  // A link into a collapsed section must open that section before scrolling.
  function reveal() {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (!target) return;
    let parent = target.parentElement;
    while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; }
    target.scrollIntoView({behavior:'instant'});
  }
  window.addEventListener('hashchange', () => {anchored = true; reveal();});
  if (location.hash) reveal();
})();
