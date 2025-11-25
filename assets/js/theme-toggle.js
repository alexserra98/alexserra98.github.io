
function setTheme(theme) {
  document.body.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  
  // Update toggle button icon if it exists
  const toggleIcon = document.getElementById('theme-toggle-icon');
  if (toggleIcon) {
    if (theme === 'dark') {
      toggleIcon.classList.remove('fa-moon');
      toggleIcon.classList.add('fa-sun');
    } else {
      toggleIcon.classList.remove('fa-sun');
      toggleIcon.classList.add('fa-moon');
    }
  }
}

function getPreferredTheme() {
  const storedTheme = localStorage.getItem('theme');
  if (storedTheme) {
    return storedTheme;
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function toggleTheme() {
  const currentTheme = document.body.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  setTheme(newTheme);
}

// Initialize theme
document.addEventListener('DOMContentLoaded', () => {
  const preferredTheme = getPreferredTheme();
  setTheme(preferredTheme);
  
  const toggleButton = document.getElementById('theme-toggle');
  if (toggleButton) {
    toggleButton.addEventListener('click', toggleTheme);
  }
});

// Also set theme immediately to avoid flash, if possible (script should be in head or early body)
const preferredTheme = getPreferredTheme();
if (document.body) {
    document.body.setAttribute('data-theme', preferredTheme);
}
