/* Color Picker Widget — injects a floating accent-color switcher into any page */
(function(){
  'use strict';

  var PALETTE = [
    { name: 'Pink',    hex: '#ed18c6', rgb: '237, 24, 198', rgb2: '161, 69, 255' },
    { name: 'Blue',    hex: '#1e90ff', rgb: '30, 144, 255',  rgb2: '99, 102, 241' },
    { name: 'Gold',    hex: '#f5a623', rgb: '245, 166, 35',  rgb2: '220, 138, 0' },
    { name: 'Emerald', hex: '#10b981', rgb: '16, 185, 129',  rgb2: '13, 148, 136' },
    { name: 'Coral',   hex: '#ff6b6b', rgb: '255, 107, 107', rgb2: '225, 29, 72' },
    { name: 'Lavender',hex: '#a855f7', rgb: '168, 85, 247',  rgb2: '124, 58, 237' },
    { name: 'Cyan',    hex: '#06b6d4', rgb: '6, 182, 212',   rgb2: '59, 130, 246' },
    { name: 'Charcoal',hex: '#6B7280', rgb: '107, 114, 128', rgb2: '75, 85, 99' },
    { name: 'Silver',  hex: '#94A3B8', rgb: '148, 163, 184', rgb2: '100, 116, 139' }
  ];

  var DEFAULT_INDEX = 3;
  var stored = sessionStorage.getItem('accent-index');
  var activeIndex = stored !== null ? parseInt(stored, 10) : DEFAULT_INDEX;
  if (activeIndex < 0 || activeIndex >= PALETTE.length) activeIndex = DEFAULT_INDEX;
  var isOpen = false;

  // Apply saved color to CSS variables immediately (before DOM paints)
  if (stored !== null) {
    var init = PALETTE[activeIndex];
    var r = document.documentElement.style;
    r.setProperty('--accent', init.hex);
    r.setProperty('--accent-rgb', init.rgb);
    r.setProperty('--accent2-rgb', init.rgb2);
  }

  // Build DOM
  var wrapper = document.createElement('div');
  wrapper.className = 'color-picker';
  wrapper.setAttribute('aria-label', 'Color theme picker');

  var panel = document.createElement('div');
  panel.className = 'color-picker-panel';
  panel.setAttribute('role', 'group');
  panel.setAttribute('aria-label', 'Color options');

  PALETTE.forEach(function(color, i){
    var swatch = document.createElement('button');
    swatch.className = 'color-picker-swatch' + (i === activeIndex ? ' active' : '');
    swatch.style.background = color.hex;
    swatch.setAttribute('aria-label', color.name + ' theme');
    swatch.setAttribute('title', color.name);
    swatch.addEventListener('click', function(e){
      e.stopPropagation();
      applyColor(i);
    });
    panel.appendChild(swatch);
  });

  var toggle = document.createElement('button');
  toggle.className = 'color-picker-toggle';
  toggle.setAttribute('aria-label', 'Open color picker');
  toggle.setAttribute('aria-expanded', 'false');
  toggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13.5" cy="6.5" r="0.5" fill="currentColor"/><circle cx="17.5" cy="10.5" r="0.5" fill="currentColor"/><circle cx="8.5" cy="7.5" r="0.5" fill="currentColor"/><circle cx="6.5" cy="12.5" r="0.5" fill="currentColor"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/></svg>';

  toggle.style.background = PALETTE[activeIndex].hex;

  toggle.addEventListener('click', function(e){
    e.stopPropagation();
    togglePanel();
  });

  wrapper.appendChild(panel);
  wrapper.appendChild(toggle);

  // Inject when DOM is ready, then sync illusion inputs with restored color
  if (document.body) {
    document.body.appendChild(wrapper);
    applyColor(activeIndex);
  } else {
    document.addEventListener('DOMContentLoaded', function(){
      document.body.appendChild(wrapper);
      applyColor(activeIndex);
    });
  }

  function togglePanel(){
    isOpen = !isOpen;
    panel.classList.toggle('open', isOpen);
    toggle.setAttribute('aria-expanded', String(isOpen));
  }

  function closePanel(){
    isOpen = false;
    panel.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
  }

  function applyColor(index){
    activeIndex = index;
    var color = PALETTE[index];
    var root = document.documentElement.style;

    root.setProperty('--accent', color.hex);
    root.setProperty('--accent-rgb', color.rgb);
    root.setProperty('--accent2-rgb', color.rgb2);

    // Update toggle button background
    toggle.style.background = color.hex;

    // Update active swatch
    var swatches = panel.querySelectorAll('.color-picker-swatch');
    for (var i = 0; i < swatches.length; i++){
      swatches[i].classList.toggle('active', i === index);
    }

    // Sync illusion color pickers that track the accent
    var accentInputs = document.querySelectorAll('input[data-accent-default]');
    for (var j = 0; j < accentInputs.length; j++){
      accentInputs[j].value = color.hex;
      accentInputs[j].dispatchEvent(new Event('input', { bubbles: true }));
    }

    // Persist choice for this browser session
    sessionStorage.setItem('accent-index', String(index));

    // Notify other scripts (e.g. canvas-based illusions)
    document.dispatchEvent(new CustomEvent('accentchange', { detail: { hex: color.hex, rgb: color.rgb, rgb2: color.rgb2 } }));
  }

  // Close on click outside
  document.addEventListener('click', function(){
    if (isOpen) closePanel();
  });

  // Close on Escape
  document.addEventListener('keydown', function(e){
    if (e.key === 'Escape' && isOpen) closePanel();
  });

  // Prevent clicks inside the widget from closing it
  wrapper.addEventListener('click', function(e){
    e.stopPropagation();
  });
})();
