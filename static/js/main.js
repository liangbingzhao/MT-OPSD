(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var IMG = 'static/images/turns/';

  var ICON_PLAY = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2.5" width="3.5" height="11" rx="1"/><rect x="9.5" y="2.5" width="3.5" height="11" rx="1"/></svg>';
  var ICON_SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var ICON_MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  /* ------------------------------------------------------------------
   * Theme toggle
   * ------------------------------------------------------------------ */
  function isDark() {
    var t = document.documentElement.dataset.theme;
    if (t) return t === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function paintThemeButtons() {
    $$('[data-theme-toggle]').forEach(function (b) { b.innerHTML = isDark() ? ICON_SUN : ICON_MOON; });
  }
  $$('[data-theme-toggle]').forEach(function (b) {
    b.addEventListener('click', function () {
      var next = isDark() ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) {}
      paintThemeButtons();
    });
  });
  paintThemeButtons();

  /* ------------------------------------------------------------------
   * Nav: show after hero, highlight current section
   * ------------------------------------------------------------------ */
  var nav = $('#nav');
  function onScroll() { nav.classList.toggle('show', window.scrollY > 320); }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  var navLinks = $$('.nav ul a');
  if ('IntersectionObserver' in window) {
    var secObs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        navLinks.forEach(function (a) { a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id); });
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    navLinks.forEach(function (a) { var s = $(a.getAttribute('href')); if (s) secObs.observe(s); });
  }

  /* ------------------------------------------------------------------
   * Reveal on scroll
   * ------------------------------------------------------------------ */
  var revealEls = $$('.reveal');
  if ('IntersectionObserver' in window) {
    var rObs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); rObs.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px' });
    revealEls.forEach(function (el) { rObs.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('in'); });
  }

  /* Placeholder links (paper / code / data not yet public) */
  $$('[data-soon]').forEach(function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); });
  });

  /* ------------------------------------------------------------------
   * Turn explorer
   * rows: [{name, ours, src(turn) -> url, degradeAt}]
   * ------------------------------------------------------------------ */
  function Explorer(root, rows) {
    this.root = root;
    this.range = $('input[type=range]', root);
    this.playBtn = $('.play', root);
    this.readout = $('.turn-readout b', root);
    this.panelsEl = $('.panels', root);
    this.ticksEl = $('.ticks', root);
    this.turn = 0;
    this.timer = null;

    var self = this;
    var ticks = '';
    for (var t = 0; t <= 10; t++) ticks += '<span data-t="' + t + '">' + t + '</span>';
    this.ticksEl.innerHTML = ticks;
    $$('span', this.ticksEl).forEach(function (s) {
      s.addEventListener('click', function () { self.stop(); self.set(+s.dataset.t); });
    });
    this.range.addEventListener('input', function () { self.stop(); self.set(+self.range.value); });
    this.playBtn.addEventListener('click', function () { self.timer ? self.stop() : self.play(); });
    this.playBtn.innerHTML = ICON_PLAY;

    this.setRows(rows);
  }
  Explorer.prototype.setRows = function (rows) {
    this.rows = rows;
    this.panelsEl.innerHTML = rows.map(function (r) {
      return '<div class="panel' + (r.ours ? ' ours' : '') + '">' +
        '<div class="label"><span class="n" title="' + r.name + '">' + r.name + '</span></div>' +
        '<div class="frame"><img alt="' + r.name + '" decoding="async"><span class="status neutral"></span></div></div>';
    }).join('');
    this.panels = $$('.panel', this.panelsEl);
    // preload every frame so scrubbing is instant
    rows.forEach(function (r) { for (var t = 0; t <= 10; t++) { var i = new Image(); i.src = r.src(t); } });
    this.set(this.turn);
  };
  Explorer.prototype.set = function (t) {
    this.turn = t;
    this.range.value = t;
    this.readout.textContent = t;
    $$('span', this.ticksEl).forEach(function (s) { s.classList.toggle('on', +s.dataset.t === t); });
    var self = this;
    this.rows.forEach(function (r, i) {
      var p = self.panels[i];
      $('img', p).src = r.src(t);
      $('img', p).alt = r.name + ', turn ' + t;
      var st = $('.status', p);
      if (t === 0) { st.className = 'status neutral'; st.textContent = 'source'; }
      else if (r.degradeAt && t >= r.degradeAt) { st.className = 'status bad'; st.textContent = 'degraded'; }
      else if (r.ours) { st.className = 'status ok'; st.textContent = 'stable'; }
      else if (r.degradeAt) { st.className = 'status neutral'; st.textContent = 'ok'; }
      else { st.className = 'status neutral'; st.textContent = ''; }
    });
  };
  Explorer.prototype.play = function () {
    var self = this;
    if (this.turn >= 10) this.set(0);
    this.playBtn.innerHTML = ICON_PAUSE;
    this.playBtn.setAttribute('aria-label', 'Pause');
    this.timer = setInterval(function () {
      if (self.turn >= 10) { self.stop(); return; }
      self.set(self.turn + 1);
    }, 850);
  };
  Explorer.prototype.stop = function () {
    clearInterval(this.timer);
    this.timer = null;
    this.playBtn.innerHTML = ICON_PLAY;
    this.playBtn.setAttribute('aria-label', 'Play');
  };

  // Teaser explorer; "degradeAt" follows the frowning-face marks in the teaser figure.
  function chef(key) { return function (t) { return IMG + 'chef_' + key + '_t' + t + '.jpg'; }; }
  var teaser = new Explorer($('#explorer-teaser'), [
    { name: 'FLUX.2-klein-base', src: chef('flux'), degradeAt: 5 },
    { name: 'FireRed-Image-Edit', src: chef('firered'), degradeAt: 3 },
    { name: 'Qwen-Image-Edit', src: chef('qwen'), degradeAt: 4 },
    { name: 'Qwen-Image-Edit + MT-OPSD', src: chef('ours'), ours: true }
  ]);

  // Auto-play the teaser once when it first scrolls into view.
  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    var played = false;
    var tObs = new IntersectionObserver(function (en) {
      if (en[0].isIntersecting && !played) { played = true; setTimeout(function () { teaser.play(); }, 600); tObs.disconnect(); }
    }, { threshold: 0.6 });
    tObs.observe($('#explorer-teaser'));
  }

  function ablRows(session) {
    function src(variant) {
      return function (t) { return IMG + (t === 0 ? 'abl_' + session + '_src' : 'abl_' + session + '_' + variant + '_t' + t) + '.jpg'; };
    }
    return [
      { name: 'MT-OPSD', src: src('ours'), ours: true },
      { name: 'w/o identity branch', src: src('noid') }
    ];
  }
  var abl = new Explorer($('#explorer-abl'), ablRows('chef'));
  $$('#explorer-abl [data-session]').forEach(function (b) {
    b.addEventListener('click', function () {
      $$('#explorer-abl [data-session]').forEach(function (x) { x.classList.toggle('on', x === b); });
      abl.setRows(ablRows(b.dataset.session));
    });
  });

  // Keyboard: arrow keys scrub whichever explorer is focused/hovered
  [teaser, abl].forEach(function (ex) {
    ex.root.addEventListener('keydown', function (e) {
      if (e.target === ex.range) return; // native handling
      if (e.key === 'ArrowRight') { ex.stop(); ex.set(Math.min(10, ex.turn + 1)); }
      if (e.key === 'ArrowLeft') { ex.stop(); ex.set(Math.max(0, ex.turn - 1)); }
    });
  });

  /* ------------------------------------------------------------------
   * Data (from the paper tables)
   * ------------------------------------------------------------------ */
  var TURNS = [3, 5, 8, 10];
  var LME = {
    proprietary: [
      ['GPT-Image-1', [0.82, 0.48, 0.09, 0.01], [0.03, 0.35, 0.63, 0.69]],
      ['GPT-Image-2', [1.00, 0.98, 0.98, 0.96], [0.00, 0.00, 0.00, 0.00]],
      ['Nano Banana', [0.67, 0.66, 0.63, 0.56], [0.00, 0.00, 0.00, 0.00]],
      ['Nano Banana Pro', [1.00, 1.00, 0.97, 0.91], [0.00, 0.00, 0.00, 0.00]]
    ],
    backbones: [
      { key: 'qwen', name: 'Qwen-Image-Edit-2511', short: 'Qwen-Image-Edit', rows: {
        base: [[0.98, 0.43, 0.08, 0.03], [0.00, 0.10, 0.45, 0.55]],
        'Emu Edit': [[0.95, 0.54, 0.19, 0.13], [0.00, 0.04, 0.21, 0.30]],
        'FreqEdit': [[0.89, 0.22, 0.03, 0.01], [0.00, 0.08, 0.41, 0.54]],
        'VAE-LFA': [[0.84, 0.46, 0.18, 0.10], [0.00, 0.05, 0.29, 0.36]],
        ours: [[1.00, 0.91, 0.63, 0.44], [0.00, 0.01, 0.02, 0.02]] } },
      { key: 'firered', name: 'FireRed-Image-Edit', short: 'FireRed-Image-Edit', rows: {
        base: [[0.95, 0.57, 0.28, 0.15], [0.01, 0.16, 0.51, 0.61]],
        'Emu Edit': [[0.97, 0.58, 0.30, 0.14], [0.01, 0.05, 0.36, 0.51]],
        'FreqEdit': [[0.93, 0.41, 0.04, 0.00], [0.01, 0.11, 0.64, 0.73]],
        'VAE-LFA': [[0.88, 0.70, 0.50, 0.37], [0.01, 0.08, 0.26, 0.33]],
        ours: [[0.98, 0.88, 0.64, 0.52], [0.00, 0.00, 0.01, 0.03]] } },
      { key: 'flux', name: 'FLUX.2-klein-base', short: 'FLUX.2-klein-base', rows: {
        base: [[0.92, 0.52, 0.29, 0.12], [0.00, 0.01, 0.17, 0.25]],
        'Emu Edit': [[0.90, 0.52, 0.23, 0.11], [0.00, 0.02, 0.12, 0.18]],
        'FreqEdit': [[0.64, 0.19, 0.03, 0.01], [0.00, 0.03, 0.35, 0.54]],
        'VAE-LFA': [[0.81, 0.51, 0.31, 0.16], [0.00, 0.03, 0.16, 0.24]],
        ours: [[0.95, 0.76, 0.57, 0.38], [0.00, 0.00, 0.02, 0.04]] } }
    ]
  };
  var BASELINES = ['Emu Edit', 'FreqEdit', 'VAE-LFA'];

  function f2(v) { return v.toFixed(2); }

  /* ------------------------------------------------------------------
   * Result cards (animated bars: base vs ours at turn 10)
   * ------------------------------------------------------------------ */
  var cards = $('#result-cards');
  cards.innerHTML = LME.backbones.map(function (b) {
    var r = b.rows;
    function row(label, cls, v) {
      return '<div class="rrow"><span class="lab">' + label + '</span><div class="bar"><i class="' + cls + '" data-w="' + v + '"></i><em data-l="' + v + '">' + f2(v) + '</em></div></div>';
    }
    return '<div class="rcard reveal"><h3>' + b.name + '</h3>' +
      row('SR@10 ↑', 'base', r.base[0][3]) + row('', 'ours', r.ours[0][3]) +
      row('CR@10 ↓', 'base', r.base[1][3]) + row('', 'ours', r.ours[1][3]) +
      '<div class="rsub"><span class="b">Base</span><span class="o">+ MT-OPSD</span></div></div>';
  }).join('');
  function growBars(el) {
    $$('.bar i', el).forEach(function (i) { i.style.width = (i.dataset.w * 100) + '%'; });
    $$('.bar em', el).forEach(function (e) { e.style.left = (e.dataset.l * 100) + '%'; });
  }
  $$('.rcard', cards).forEach(function (c) {
    if ('IntersectionObserver' in window) {
      var o = new IntersectionObserver(function (en) {
        if (en[0].isIntersecting) { c.classList.add('in'); setTimeout(function () { growBars(c); }, 150); o.disconnect(); }
      }, { rootMargin: '0px 0px -10% 0px' });
      o.observe(c);
    } else { c.classList.add('in'); growBars(c); }
  });

  /* ------------------------------------------------------------------
   * Line charts (hand-rolled SVG, crosshair + tooltip)
   * ------------------------------------------------------------------ */
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function drawChart(container, bb, metric) {
    var old = $('svg', container); if (old) old.remove();
    var oldTip = $('.tip', container); if (oldTip) oldTip.remove();
    var mi = metric === 'sr' ? 0 : 1;
    var W = 520, H = 290, m = { l: 38, r: 64, t: 16, b: 30 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var x = function (t) { return m.l + (t - 1) / 9 * iw; };
    var y = function (v) { return m.t + (1 - v) * ih; };

    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
      'aria-label': (metric === 'sr' ? 'Success rate' : 'Collapse rate') + ' over turns for ' + bb.name });
    var g = el('g', { 'class': 'grid' }, svg);
    var ax = el('g', { 'class': 'axis' }, svg);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      el('line', { x1: m.l, x2: m.l + iw, y1: y(v), y2: y(v) }, g);
      el('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end' }, ax).textContent = v.toFixed(2).replace(/0$/, '');
    });
    TURNS.forEach(function (t) {
      el('text', { x: x(t), y: H - 8, 'text-anchor': 'middle' }, ax).textContent = 'T' + t;
    });

    function path(vals) { return vals.map(function (v, i) { return (i ? 'L' : 'M') + x(TURNS[i]) + ',' + y(v); }).join(''); }

    BASELINES.forEach(function (k) { el('path', { d: path(bb.rows[k][mi]), 'class': 'ctx' }, svg); });
    var series = [
      { key: 'base', name: 'Base', color: 'var(--series-base)' },
      { key: 'ours', name: '+ MT-OPSD', color: 'var(--series-ours)' }
    ];
    series.forEach(function (s) {
      var vals = bb.rows[s.key][mi];
      var p = el('path', { d: path(vals), 'class': 'ln', stroke: s.color }, svg);
      s.path = p;
      vals.forEach(function (v, i) { el('circle', { cx: x(TURNS[i]), cy: y(v), r: 4.5, fill: s.color, 'class': 'dot' }, svg); });
      // direct label at end of the line
      var last = vals[vals.length - 1];
      s.endY = y(last);
      s.label = el('text', { x: x(10) + 10, y: y(last) + 4, 'class': 'dlab' }, svg);
      s.label.textContent = f2(last);
    });
    // de-collide end labels
    var a = series[0], b = series[1];
    if (Math.abs(a.endY - b.endY) < 14) {
      var mid = (a.endY + b.endY) / 2, up = a.endY < b.endY ? a : b, dn = up === a ? b : a;
      up.label.setAttribute('y', mid - 7 + 4); dn.label.setAttribute('y', mid + 7 + 4);
    }

    var hair = el('line', { 'class': 'hair', y1: m.t, y2: m.t + ih, x1: -10, x2: -10, opacity: 0 }, svg);
    var hit = el('rect', { x: m.l - 20, y: 0, width: iw + 40, height: H, fill: 'transparent' }, svg);
    container.appendChild(svg);

    // draw-in animation
    series.forEach(function (s) {
      var L = s.path.getTotalLength();
      s.path.style.strokeDasharray = L;
      s.path.style.strokeDashoffset = L;
      s.path.getBoundingClientRect();
      s.path.style.transition = 'stroke-dashoffset 1s cubic-bezier(.2,.8,.2,1)';
      s.path.style.strokeDashoffset = 0;
    });

    var tip = document.createElement('div');
    tip.className = 'tip';
    container.appendChild(tip);
    var allRows = [['+ MT-OPSD', 'ours', 'var(--series-ours)'], ['Base', 'base', 'var(--series-base)']]
      .concat(BASELINES.map(function (k) { return [k, k, 'var(--series-ctx)']; }));

    function move(ev) {
      var rect = svg.getBoundingClientRect();
      var px = (ev.clientX - rect.left) / rect.width * W;
      var best = 0, bd = 1e9;
      TURNS.forEach(function (t, i) { var d = Math.abs(x(t) - px); if (d < bd) { bd = d; best = i; } });
      var tx = x(TURNS[best]);
      hair.setAttribute('x1', tx); hair.setAttribute('x2', tx); hair.setAttribute('opacity', 1);
      var rowsHtml = allRows.map(function (r) {
        return '<div class="r"><span><i style="background:' + r[2] + '"></i>' + r[0] + '</span><b>' + f2(bb.rows[r[1]][mi][best]) + '</b></div>';
      }).join('');
      tip.innerHTML = '<div class="h">' + bb.short + ' · ' + (metric === 'sr' ? 'SR' : 'CR') + '@' + TURNS[best] + '</div>' + rowsHtml;
      var cRect = container.getBoundingClientRect();
      var left = rect.left - cRect.left + tx / W * rect.width;
      var tw = tip.offsetWidth;
      left = left + 14 + tw > cRect.width ? left - tw - 14 : left + 14;
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = (rect.top - cRect.top + 30) + 'px';
      tip.classList.add('on');
    }
    function leave() { tip.classList.remove('on'); hair.setAttribute('opacity', 0); }
    hit.addEventListener('pointermove', move);
    hit.addEventListener('pointerdown', move);
    hit.addEventListener('pointerleave', leave);
  }

  var seg = $('#bb-seg');
  seg.innerHTML = LME.backbones.map(function (b, i) {
    return '<button data-i="' + i + '"' + (i === 0 ? ' class="on"' : '') + '>' + b.short.replace('-Image-Edit', '').replace('-base', '') + '</button>';
  }).join('');
  function renderCharts(i) {
    $$('#chart-card .chart').forEach(function (c) { drawChart(c, LME.backbones[i], c.dataset.metric); });
  }
  $$('button', seg).forEach(function (b) {
    b.addEventListener('click', function () {
      $$('button', seg).forEach(function (x) { x.classList.toggle('on', x === b); });
      renderCharts(+b.dataset.i);
    });
  });
  var chartDrawn = false;
  if ('IntersectionObserver' in window) {
    var cObs = new IntersectionObserver(function (en) {
      if (en[0].isIntersecting && !chartDrawn) { chartDrawn = true; renderCharts(0); cObs.disconnect(); }
    }, { rootMargin: '0px 0px -15% 0px' });
    cObs.observe($('#chart-card'));
  } else { renderCharts(0); }

  /* ------------------------------------------------------------------
   * Gallery tabs
   * ------------------------------------------------------------------ */
  [['#gal-seg', '.gallery .pane'], ['#tbl-seg', '.tables .pane']].forEach(function (g) {
    $$(g[0] + ' button').forEach(function (b) {
      b.addEventListener('click', function () {
        $$(g[0] + ' button').forEach(function (x) { x.classList.toggle('on', x === b); });
        $$(g[1]).forEach(function (p) { p.classList.toggle('on', p.id === b.dataset.pane); });
      });
    });
  });

  /* ------------------------------------------------------------------
   * Lightbox
   * ------------------------------------------------------------------ */
  var lb = $('#lightbox'), lbImg = $('img', lb);
  $$('.figure img, .donut img').forEach(function (img) {
    img.addEventListener('click', function () {
      lbImg.src = img.currentSrc || img.src; lbImg.alt = img.alt;
      lb.classList.add('on'); document.body.style.overflow = 'hidden';
    });
  });
  function closeLb() { lb.classList.remove('on'); document.body.style.overflow = ''; }
  lb.addEventListener('click', closeLb);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeLb(); });

  /* ------------------------------------------------------------------
   * BibTeX copy
   * ------------------------------------------------------------------ */
  $('#copy-bib').addEventListener('click', function () {
    var btn = this, text = $('#bib-text').textContent;
    function done() { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1600); }
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
    else { var r = document.createRange(); r.selectNodeContents($('#bib-text')); var s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand('copy'); done(); }
  });

  /* ------------------------------------------------------------------
   * KaTeX
   * ------------------------------------------------------------------ */
  function renderMath() {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, { delimiters: [{ left: '$$', right: '$$', display: true }], throwOnError: false });
    }
  }
  if (document.readyState === 'complete') renderMath(); else window.addEventListener('load', renderMath);
})();
