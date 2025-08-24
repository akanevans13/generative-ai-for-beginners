(function () {
	const navToggleButton = document.querySelector('.nav-toggle');
	const siteNav = document.getElementById('site-nav');

	function toggleNav() {
		const isExpanded = navToggleButton.getAttribute('aria-expanded') === 'true';
		navToggleButton.setAttribute('aria-expanded', String(!isExpanded));
		siteNav.style.display = !isExpanded ? 'block' : '';
	}

	if (navToggleButton) {
		navToggleButton.addEventListener('click', toggleNav);
	}

	// Smooth scroll
	document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
		anchor.addEventListener('click', function (event) {
			const targetId = this.getAttribute('href');
			if (targetId && targetId.length > 1) {
				event.preventDefault();
				const target = document.querySelector(targetId);
				if (target) {
					target.scrollIntoView({ behavior: 'smooth' });
				}
			}
		});
	});

	// Simple lightbox
	const gallery = document.querySelector('[data-lightbox]');
	if (gallery) {
		gallery.addEventListener('click', function (event) {
			const link = event.target.closest('a');
			if (!link) return;
			event.preventDefault();
			openLightbox(link.getAttribute('href') || '', link.dataset.title || '');
		});
	}

	function openLightbox(src, caption) {
		const backdrop = document.createElement('div');
		backdrop.className = 'lightbox-backdrop';
		backdrop.setAttribute('role', 'dialog');
		backdrop.setAttribute('aria-modal', 'true');
		backdrop.addEventListener('click', closeLightbox);

		const figure = document.createElement('figure');
		figure.className = 'lightbox-figure';

		const img = document.createElement('img');
		img.className = 'lightbox-img';
		img.alt = caption || 'Artwork preview';
		img.src = src;

		const figcaption = document.createElement('figcaption');
		figcaption.className = 'lightbox-caption';
		figcaption.textContent = caption;

		figure.appendChild(img);
		if (caption) figure.appendChild(figcaption);
		backdrop.appendChild(figure);
		document.body.appendChild(backdrop);

		document.addEventListener('keydown', escToClose);
		function escToClose(e) { if (e.key === 'Escape') closeLightbox(); }
		function closeLightbox() {
			document.removeEventListener('keydown', escToClose);
			backdrop.removeEventListener('click', closeLightbox);
			backdrop.remove();
		}
	}

	// Footer year
	const yearEl = document.getElementById('year');
	if (yearEl) yearEl.textContent = String(new Date().getFullYear());
})();