/**
 * API 및 정적 자원 URL 설정 (로컬/시놀로지 공통)
 *
 * REACT_APP_API_URL:
 * - 비설정 또는 '' : 같은 출처 사용
 *   - 로컬: /api → setupProxy → localhost:5000 (Express). FastAPI(ai_server 8000)로 두지 말 것.
 *   - 시놀로지: /api → nginx → materiai_backend:5000
 * - 'https://materiai.example.com' : 별도 도메인 지정 시
 * - 'https://materiai.example.com/api' : API 풀 경로 지정 시
 */
const raw = (process.env.REACT_APP_API_URL || '').trim();

/** API 요청용 베이스 (예: '/api' 또는 'https://x.com/api') */
export function getApiBase() {
  if (!raw) return '/api';
  const base = raw.replace(/\/+$/, '');
  return base.includes('/api') ? base : `${base}/api`;
}

/** 정적 자원(이미지, 업로드)용 출처 (예: '' 또는 'https://x.com') */
export function getOrigin() {
  if (!raw) return '';
  const base = raw.replace(/\/+$/, '');
  const idx = base.indexOf('/api');
  return idx > 0 ? base.slice(0, idx) : base;
}

/**
 * 파일 업로드·서빙 전용 출처
 * REACT_APP_FILE_URL 이 설정된 경우 그 값 사용 (로컬 개발에서 배포 서버로 파일 전송 시)
 * 미설정이면 getOrigin() 과 동일
 */
export function getFileOrigin() {
  const fileRaw = (process.env.REACT_APP_FILE_URL || '').trim();
  if (fileRaw) return fileRaw.replace(/\/+$/, '');
  return getOrigin();
}
