css_scroll_test = """
@keyframes scrollFadeIn {
    0% {
        opacity: 0.2;
        transform: translateY(35px) scale(0.98);
        filter: blur(2px);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
}

@supports (animation-timeline: view()) {
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        animation: scrollFadeIn linear both !important;
        animation-timeline: view() !important;
        animation-range: entry 15% cover 35% !important;
    }
    .scoreboard-card {
        animation: scrollFadeIn linear both !important;
        animation-timeline: view() !important;
        animation-range: entry 15% cover 35% !important;
    }
}
"""
print("CSS Scroll animations defined!")
