import { test, expect } from '../../fixtures/test';
import { addToCart } from '../../helpers/api';
import { P, USERS } from '../../helpers/data';
import { meta } from '../../helpers/meta';
import { expectFlash } from '../../helpers/ui';

test(
  '[AUTH-01] registers a new customer with normalized name and email',
  meta({
    modules: ['auth'],
    crit: 'high',
    covers: ['auth/logic.ts', 'auth/routes.ts', 'auth/service.ts', 'auth/views/register.ejs', 'shared/session.ts'],
  }),
  async ({ page }) => {
    await page.goto('/catalog');
    await page.getByTestId('nav-register').click();
    await page.getByTestId('register-name').fill('  Dana Kovalenko ');
    await page.getByTestId('register-email').fill(' Dana.K@Example.COM ');
    await page.getByTestId('register-password').fill('secret12');
    await page.getByTestId('register-confirm-password').fill('secret12');
    await page.getByTestId('register-submit').click();

    await expect(page).toHaveURL(/\/catalog$/);
    await expectFlash(page, 'Welcome, Dana Kovalenko! Your account has been created.');
    await expect(page.getByTestId('user-name')).toHaveText('Dana Kovalenko');
    await expect(page.getByTestId('nav-orders')).toBeVisible();
    await expect(page.getByTestId('nav-login')).toHaveCount(0);

    // The account is stored with a lower-cased email: signing in again with it works.
    await page.getByTestId('nav-logout').click();
    await expectFlash(page, 'You have been signed out');
    await page.getByTestId('nav-login').click();
    await page.getByTestId('login-email').fill('dana.k@example.com');
    await page.getByTestId('login-password').fill('secret12');
    await page.getByTestId('login-submit').click();
    await expectFlash(page, 'Signed in as Dana Kovalenko');
    await expect(page.getByTestId('user-name')).toHaveText('Dana Kovalenko');
  },
);

test(
  '[AUTH-02] shows registration errors for every invalid field and a taken email',
  meta({
    modules: ['auth'],
    crit: 'medium',
    covers: ['auth/logic.ts', 'auth/routes.ts', 'auth/service.ts', 'auth/views/register.ejs'],
  }),
  async ({ page }) => {
    await page.goto('/register');
    await page.getByTestId('register-name').fill('A');
    await page.getByTestId('register-email').fill('not-an-email');
    await page.getByTestId('register-password').fill('password');
    await page.getByTestId('register-confirm-password').fill('password1');
    await page.getByTestId('register-submit').click();

    await expect(page.getByTestId('register-name-error')).toHaveText('Name must be at least 2 characters');
    await expect(page.getByTestId('register-email-error')).toHaveText('Enter a valid email address');
    await expect(page.getByTestId('register-password-error')).toHaveText(
      'Password must contain at least one letter and one digit',
    );
    await expect(page.getByTestId('register-confirm-password-error')).toHaveText('Passwords do not match');
    await expect(page.getByTestId('register-name')).toHaveValue('A');
    await expect(page.getByTestId('register-email')).toHaveValue('not-an-email');

    // Second attempt: valid 2-character name, existing email (other case), too short password.
    await page.getByTestId('register-name').fill('Al');
    await page.getByTestId('register-email').fill('ALICE@example.com');
    await page.getByTestId('register-password').fill('abc1');
    await page.getByTestId('register-confirm-password').fill('abc1');
    await page.getByTestId('register-submit').click();

    await expect(page.getByTestId('register-email-error')).toHaveText('Email is already registered');
    await expect(page.getByTestId('register-password-error')).toHaveText('Password must be at least 8 characters');
    await expect(page.getByTestId('register-name-error')).toHaveCount(0);
    await expect(page.getByTestId('register-confirm-password-error')).toHaveCount(0);
    await expect(page.getByTestId('nav-login')).toBeVisible();
  },
);

test(
  '[AUTH-03] rejects invalid credentials and returns to the requested page after sign-in',
  meta({
    modules: ['auth'],
    crit: 'high',
    covers: ['auth/logic.ts', 'auth/routes.ts', 'auth/service.ts', 'auth/views/login.ejs', 'shared/middleware.ts', 'shared/password.ts'],
  }),
  async ({ page }) => {
    await page.goto('/orders');
    await expect(page).toHaveURL(/\/login\?next=%2Forders$/);

    await page.getByTestId('login-submit').click();
    await expect(page.getByTestId('login-email-error')).toHaveText('Email is required');
    await expect(page.getByTestId('login-password-error')).toHaveText('Password is required');

    await page.getByTestId('login-email').fill(USERS.alice.email);
    await page.getByTestId('login-password').fill('Password124');
    await page.getByTestId('login-submit').click();
    await expect(page.getByTestId('login-error')).toHaveText('Invalid email or password');
    await expect(page.getByTestId('login-email')).toHaveValue(USERS.alice.email);
    await expect(page.getByTestId('login-email-error')).toHaveCount(0);

    await page.getByTestId('login-email').fill('nobody@example.com');
    await page.getByTestId('login-password').fill(USERS.alice.password);
    await page.getByTestId('login-submit').click();
    await expect(page.getByTestId('login-error')).toHaveText('Invalid email or password');
    await expect(page.getByTestId('nav-login')).toBeVisible();

    await page.getByTestId('login-email').fill(USERS.alice.email);
    await page.getByTestId('login-password').fill(USERS.alice.password);
    await page.getByTestId('login-submit').click();
    await expect(page).toHaveURL(/\/orders$/);
    await expectFlash(page, 'Signed in as Alice Johnson');
    await expect(page.getByTestId('user-name')).toHaveText('Alice Johnson');
  },
);

test(
  '[AUTH-04] signs out and drops the cart of the session',
  meta({
    modules: ['auth', 'cart'],
    crit: 'medium',
    covers: ['auth/routes.ts', 'shared/session.ts', 'shared/views/header.ejs', 'cart/routes.ts'],
  }),
  async ({ page, loginAs }) => {
    await loginAs(USERS.bob);
    await addToCart(page.request, P.yogaMat.id, 2);
    await page.goto('/cart');
    await expect(page.getByTestId('cart-count')).toHaveText('2');
    await expect(page.getByTestId('user-name')).toHaveText('Bob Smith');

    await page.getByTestId('nav-logout').click();
    await expectFlash(page, 'You have been signed out');
    await expect(page).toHaveURL(/\/catalog$/);
    await expect(page.getByTestId('cart-count')).toHaveText('0');
    await expect(page.getByTestId('user-name')).toHaveCount(0);
    await expect(page.getByTestId('nav-login')).toBeVisible();

    await page.getByTestId('nav-cart').click();
    await expect(page.getByTestId('cart-empty')).toBeVisible();
    await page.goto('/checkout');
    await expect(page).toHaveURL(/\/login\?next=%2Fcheckout$/);
  },
);
