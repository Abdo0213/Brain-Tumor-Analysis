document.addEventListener('DOMContentLoaded', () => {
    // Animate elements when they come into view
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.custom-box, .stImage, .stButton');
        elements.forEach(el => {
            const elTop = el.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            if (elTop < windowHeight - 100) {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }
        });
    };
    
    // Set initial state
    const animatedElements = document.querySelectorAll('.custom-box, .stImage, .stButton');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'all 0.6s ease-out';
    });
    
    // Trigger on load and scroll
    animateOnScroll();
    window.addEventListener('scroll', animateOnScroll);
});