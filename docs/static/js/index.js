function copyBibTeX() {
    var bibtexElement = document.getElementById('bibtex-code');
    var button = document.querySelector('.copy-bibtex-btn');
    var copyText = button.querySelector('.copy-text');

    if (!bibtexElement) return;

    navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
        button.classList.add('copied');
        copyText.textContent = 'Copied!';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    }).catch(function() {
        var textArea = document.createElement('textarea');
        textArea.value = bibtexElement.textContent;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        button.classList.add('copied');
        copyText.textContent = 'Copied!';
        setTimeout(function() {
            button.classList.remove('copied');
            copyText.textContent = 'Copy';
        }, 2000);
    });
}

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function() {
    var scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});
