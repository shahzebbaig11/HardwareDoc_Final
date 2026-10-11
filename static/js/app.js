/**
 * HardwareDoc — Main JavaScript
 * Handles: mobile sidebar, flash message auto-dismiss, answer form safety
 */

(function () {
  'use strict';

  // ── Mobile sidebar toggle ────────────────────────────────
  const sidebar  = document.getElementById('sidebar');
  const toggle   = document.getElementById('menu-toggle');
  const overlay  = document.getElementById('overlay');

  function openSidebar() {
    if (!sidebar) return;
    sidebar.classList.add('open');
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
    if (toggle) toggle.setAttribute('aria-expanded', 'true');
  }

  function closeSidebar() {
    if (!sidebar) return;
    sidebar.classList.remove('open');
    overlay.classList.remove('visible');
    document.body.style.overflow = '';
    if (toggle) toggle.setAttribute('aria-expanded', 'false');
  }

  if (toggle) {
    toggle.addEventListener('click', function () {
      if (sidebar.classList.contains('open')) {
        closeSidebar();
      } else {
        openSidebar();
      }
    });
  }

  if (overlay) {
    overlay.addEventListener('click', closeSidebar);
  }

  // Close sidebar on nav link click (mobile)
  if (sidebar) {
    sidebar.querySelectorAll('.nav-link').forEach(function (link) {
      link.addEventListener('click', function () {
        if (window.innerWidth <= 900) {
          closeSidebar();
        }
      });
    });
  }

  // ── Flash message auto-dismiss ───────────────────────────
  document.querySelectorAll('.flash').forEach(function (flash) {
    setTimeout(function () {
      flash.style.transition = 'opacity 0.4s';
      flash.style.opacity = '0';
      setTimeout(function () { flash.remove(); }, 400);
    }, 5000);
  });

  // ── Answer form: prevent double submission safely ────────
  // IMPORTANT: We must NOT disable submit buttons synchronously inside the
  // submit event — doing so causes Chrome/Edge to drop the clicked button's
  // name=value from the POST payload, which makes answer="" → "Please select".
  // Solution: defer the disable with setTimeout(0) so the browser has already
  // captured the payload before the buttons are disabled.
  document.querySelectorAll('form').forEach(function (form) {
    var submitted = false;
    form.addEventListener('submit', function (e) {
      if (submitted) {
        // Block duplicate submissions entirely
        e.preventDefault();
        return;
      }
      submitted = true;

      // Defer disabling until after the browser has read the submit button value
      setTimeout(function () {
        form.querySelectorAll('button[type="submit"]').forEach(function (btn) {
          btn.disabled = true;
          btn.style.opacity = '0.6';
        });
      }, 0);
    });
  });

})();
