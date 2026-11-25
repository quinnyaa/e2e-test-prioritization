import { Router } from 'express';
import { setFlash, setSessionUser, startNewSession } from '../../shared/session';
import { hasErrors, safeRedirectPath, validateLogin, validateRegistration } from './logic';
import { authenticate, createUser, emailExists } from './service';

export const authRouter = Router();

function field(body: Record<string, unknown>, name: string): string {
  const value = body[name];
  return typeof value === 'string' ? value : '';
}

authRouter.get('/login', (req, res) => {
  if (req.user) {
    res.redirect('/catalog');
    return;
  }
  res.render('modules/auth/views/login', {
    title: 'Sign in',
    values: { email: '' },
    errors: {},
    next: safeRedirectPath(req.query.next),
  });
});

authRouter.post('/login', (req, res) => {
  const input = { email: field(req.body, 'email'), password: field(req.body, 'password') };
  const next = safeRedirectPath(req.body.next);
  const errors: Record<string, string> = validateLogin(input);
  let user = null;
  if (!hasErrors(errors)) {
    user = authenticate(input.email, input.password);
    if (!user) errors.form = 'Invalid email or password';
  }
  if (!user) {
    res.status(400).render('modules/auth/views/login', {
      title: 'Sign in',
      values: { email: input.email },
      errors,
      next,
    });
    return;
  }
  setSessionUser(req, user.id);
  setFlash(req, 'success', `Signed in as ${user.name}`);
  res.redirect(next);
});

authRouter.get('/register', (req, res) => {
  if (req.user) {
    res.redirect('/catalog');
    return;
  }
  res.render('modules/auth/views/register', {
    title: 'Create account',
    values: { name: '', email: '' },
    errors: {},
  });
});

authRouter.post('/register', (req, res) => {
  const input = {
    name: field(req.body, 'name'),
    email: field(req.body, 'email'),
    password: field(req.body, 'password'),
    confirmPassword: field(req.body, 'confirmPassword'),
  };
  const errors: Record<string, string> = validateRegistration(input);
  if (!errors.email && emailExists(input.email)) errors.email = 'Email is already registered';
  if (hasErrors(errors)) {
    res.status(400).render('modules/auth/views/register', {
      title: 'Create account',
      values: { name: input.name, email: input.email },
      errors,
    });
    return;
  }
  const userId = createUser(input.name, input.email, input.password);
  setSessionUser(req, userId);
  setFlash(req, 'success', `Welcome, ${input.name.trim()}! Your account has been created.`);
  res.redirect('/catalog');
});

authRouter.post('/logout', (req, res) => {
  startNewSession(req, res);
  setFlash(req, 'success', 'You have been signed out');
  res.redirect('/catalog');
});
