if (!crossOriginIsolated && 'serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) {
  navigator.serviceWorker.register(new URL('sw.js', location.href).pathname).then(() => navigator.serviceWorker.ready).then(() => {
    if (!navigator.serviceWorker.controller || !sessionStorage.getItem('coi-reloaded')) {
      if (!sessionStorage.getItem('coi-reloaded')) { sessionStorage.setItem('coi-reloaded', '1'); location.reload(); }
    }
  });
}
