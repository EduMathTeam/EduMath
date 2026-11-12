const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function request(path, signal) {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) throw new Error(`Ошибка сервера: HTTP ${response.status}`);
  const data = await response.json();
  if (!Array.isArray(data)) throw new Error('Неожиданный формат ответа API: ожидался массив');
  return data;
}

export const getCourses = (signal) => request('/courses', signal);
export const getRecommendations = (userId, signal) => request(`/recommendations?user_id=${encodeURIComponent(userId)}`, signal);
export const apiBase = API_BASE;
