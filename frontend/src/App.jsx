import React, { useEffect, useMemo, useState } from 'react';
import { apiBase, getCourses, getRecommendations } from './api';

const icons = { 'Программирование': '⌘', 'Математика': '∑', 'Дизайн': '✦', 'Анализ данных': '▥', 'Иностранные языки': '文' };
const gradients = ['purple','blue','peach','mint','pink'];

function CourseCard({ course, index, recommended = false }) {
  const category = course.category || 'Образование';
  const score = typeof course.score === 'number' ? course.score : null;
  return <article className="course-card">
    <div className={`course-art ${gradients[index % gradients.length]}`}>
      <span className="art-icon">{icons[category] || '✳'}</span>
      <span className="art-label">EDUMATH / COURSE</span>
    </div>
    <div className="course-content">
      <div className="course-meta"><span>{category}</span>{recommended && <span className="rec-tag">✦ Для вас</span>}</div>
      <h3>{course.title || 'Без названия'}</h3>
      <div className="course-bottom"><span>Курс · #{String(course.course_id ?? '—')}</span>{score !== null && <span className="score">Оценка модели: {score.toFixed(3)}</span>}</div>
    </div>
  </article>;
}

export default function App() {
  const [tab, setTab] = useState('catalog');
  const [courses, setCourses] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [userId, setUserId] = useState('1');
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('Все направления');
  const [loadingCourses, setLoadingCourses] = useState(true);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [courseError, setCourseError] = useState('');
  const [recError, setRecError] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoadingCourses(true);
    setCourseError('');
    getCourses(controller.signal)
      .then(setCourses)
      .catch(error => { if (error.name !== 'AbortError') setCourseError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingCourses(false); });
    return () => controller.abort();
  }, [refreshKey]);

  useEffect(() => {
    if (tab !== 'recommended') return;
    const controller = new AbortController();
    setLoadingRecommendations(true);
    setRecError('');
    getRecommendations(userId, controller.signal)
      .then(setRecommendations)
      .catch(error => { if (error.name !== 'AbortError') setRecError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingRecommendations(false); });
    return () => controller.abort();
  }, [tab, userId, refreshKey]);

  const categories = useMemo(() => ['Все направления', ...new Set(courses.map(c => c.category).filter(Boolean))], [courses]);
  const shownCourses = useMemo(() => courses.filter(c =>
    (category === 'Все направления' || c.category === category) &&
    String(c.title || '').toLocaleLowerCase('ru').includes(search.toLocaleLowerCase('ru').trim())
  ), [courses, search, category]);

  const items = tab === 'catalog' ? shownCourses : recommendations;
  const loading = tab === 'catalog' ? loadingCourses : loadingRecommendations;
  const error = tab === 'catalog' ? courseError : recError;

  return <div className="layout">
    <aside className="sidebar">
      <div className="brand"><span className="brand-symbol">✳</span><span>Edu<span className="brand-accent">Math</span></span></div>
      <p className="side-label">ОБУЧЕНИЕ</p>
      <button className={`nav-link ${tab === 'catalog' ? 'active' : ''}`} onClick={() => setTab('catalog')}><span>▦</span> Каталог курсов</button>
      <button className={`nav-link ${tab === 'recommended' ? 'active' : ''}`} onClick={() => setTab('recommended')}><span>✦</span> Рекомендации</button>
      <div className="sidebar-footer"><div className="side-avatar">E</div><div><strong>EduMath</strong><small>Учебный прототип</small></div></div>
    </aside>

    <main className="main">
      <header className="topbar"><span>Платформа образовательных материалов</span><span className="api-status"><span className="status-dot" /> API: {apiBase}</span></header>
      <div className="content">
        {tab === 'catalog' ? <>
          <section className="hero"><div><span className="eyebrow">✦ ТВОЙ ПУТЬ К ЗНАНИЯМ</span><h1>Учись тому, что<br/><em>действительно интересно.</em></h1><p>Изучай образовательные курсы и находи новые направления для развития.</p><button className="primary" onClick={() => setTab('recommended')}>Мои рекомендации <span>↗</span></button></div><div className="hero-figure" aria-hidden="true"><span>✳</span><i className="figure-orbit one"/><i className="figure-orbit two"/></div></section>
          <div className="section-heading"><div><span className="eyebrow dark">ОТКРЫВАЙ НОВОЕ</span><h2>Каталог курсов</h2><p>Все доступные образовательные материалы в одном месте</p></div><span className="count-pill">{shownCourses.length} курсов</span></div>
          <div className="filters"><div className="search-wrap"><span>⌕</span><input aria-label="Поиск курса" placeholder="Поиск по названию курса..." value={search} onChange={e => setSearch(e.target.value)}/></div><select aria-label="Категория" value={category} onChange={e => setCategory(e.target.value)}>{categories.map(c => <option key={c}>{c}</option>)}</select></div>
        </> : <>
          <section className="rec-hero"><span className="eyebrow dark">✦ ПЕРСОНАЛЬНАЯ ПОДБОРКА</span><h1>Рекомендации для тебя</h1><p>Подборка курсов, полученная от рекомендательного API.</p><div className="user-control"><label htmlFor="userId">ID пользователя</label><input id="userId" type="number" min="1" step="1" value={userId} onChange={e => setUserId(e.target.value)} /><span>Сейчас рекомендации тестовые и могут совпадать у разных пользователей.</span></div></section>
          <div className="section-heading"><div><h2>Твоя подборка</h2><p>На основе ответа сервера</p></div><span className="count-pill">{recommendations.length} курсов</span></div>
        </>}
        {loading && <div className="notice">Загружаем курсы...</div>}
        {!loading && error && <div className="notice error"><strong>Не удалось подключиться к API.</strong><p>{error}</p><p>Проверь, что FastAPI запущен на {apiBase} и CORS разрешает адрес сайта.</p><button className="secondary" onClick={() => setRefreshKey(k => k + 1)}>Повторить</button></div>}
        {!loading && !error && items.length === 0 && <div className="notice">Ничего не найдено. Попробуй изменить фильтры или пользователя.</div>}
        {!loading && !error && items.length > 0 && <section className="course-grid">{items.map((course, index) => <CourseCard key={`${course.course_id}-${index}`} course={course} index={index} recommended={tab === 'recommended'}/>)}</section>}
        <footer className="footer">EduMath · Учебный проект рекомендательной системы · 2026</footer>
      </div>
    </main>
  </div>;
}
