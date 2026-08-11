
let isLogin = false;
const API = CONFIG.API_URL;

// Toggle Login/Register
document.getElementById('toggleAuth').onclick = (e) => {
  e.preventDefault();
  isLogin = !isLogin;
  document.getElementById('authTitle').innerText = isLogin ? "Log in to TermPaper" : "Create a TermPaper account";
  document.getElementById('authBtn').innerText = isLogin ? "Log in" : "Create account";
  document.getElementById('usernameField').classList.toggle('hidden');
}

document.getElementById('authForm').onsubmit = async (e) => {
  e.preventDefault();
  const url = isLogin ? '/api/login' : '/api/register';
  const body = isLogin 
    ? { login: document.getElementById('email').value, password: document.getElementById('password').value }
    : { username: document.getElementById('username').value, email: document.getElementById('email').value, password: document.getElementById('password').value };

  const res = await fetch(API + url, { method: 'POST', headers: {'Content-Type':'application/json'}, credentials: 'include', body: JSON.stringify(body) });
  const data = await res.json();
  if(res.ok) checkAuth(); else alert(data.error);
}

async function checkAuth(){
  const res = await fetch(API + '/api/me', { credentials: 'include' });
  if(res.ok){
    const user = await res.json();
    document.getElementById('authSection').classList.add('hidden');
    document.getElementById('appSection').classList.remove('hidden');
    document.getElementById('userInfo').innerText = `Hi, ${user.username}`;
    document.getElementById('viewCount').innerText = user.views;
    if(user.is_admin) document.getElementById('adminBtn').classList.remove('hidden');
  }
}

document.getElementById('paperForm').onsubmit = async (e) => {
  e.preventDefault();
  document.getElementById('loader').classList.remove('hidden');
  const topic = document.getElementById('topic').value;
  const res = await fetch(API + '/api/generate', { method: 'POST', headers: {'Content-Type':'application/json'}, credentials: 'include', body: JSON.stringify({topic}) });
  const data = await res.json();
  document.getElementById('resultTitle').innerText = `Paper: ${topic}`;
  document.getElementById('resultContent').innerText = data.content;
  document.getElementById('viewCount').innerText = data.views;
  document.getElementById('result').classList.remove('hidden');
  document.getElementById('loader').classList.add('hidden');
}

async function showAdmin(){
  const res = await fetch(API + '/api/admin/users', { credentials: 'include' });
  const users = await res.json();
  let html = '<tr><th>ID</th><th>User</th><th>Email</th><th>Views</th><th>Action</th></tr>';
  users.forEach(u => {
    html += `<tr><td>${u.id}</td><td>${u.username}</td><td>${u.email}</td><td>${u.views}</td><td><button onclick="deleteUser(${u.id})" class="btn-danger">Delete</button></td></tr>`
  });
  document.getElementById('usersTable').innerHTML = html;
  document.getElementById('adminPanel').classList.remove('hidden');
}

async function deleteUser(id){
  if(confirm('Delete this user?')){
    await fetch(`${API}/api/admin/user/${id}`, { method: 'DELETE', credentials: 'include' });
    showAdmin();
  }
}

async function logout(){
  await fetch(API + '/api/logout', { credentials: 'include' });
  location.reload();
}

function togglePassword(){
  const p = document.getElementById('password');
  p.type = p.type === 'password' ? 'text' : 'password';
}

checkAuth(); // run on load
