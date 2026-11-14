scroll_test = """
<style>
    /* Scroll Reveal Transition Animations */
    .scroll-reveal {
        opacity: 0;
        transform: translateY(25px);
        transition: opacity 0.65s cubic-bezier(0.16, 1, 0.3, 1), transform 0.65s cubic-bezier(0.16, 1, 0.3, 1);
        will-change: opacity, transform;
    }
    .scroll-reveal.visible {
        opacity: 1;
        transform: translateY(0);
    }

    /* Container Card Enhancements */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.35s ease, border-color 0.35s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(76, 175, 80, 0.8) !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.8), 0 0 25px rgba(76,175,80,0.3) !important;
    }
</style>
<script>
    // IntersectionObserver to add .visible class when user scrolls
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.15 });

    document.querySelectorAll('.scroll-reveal').forEach(el => observer.observe(el));
</script>
"""
print("Scroll test script created!")
