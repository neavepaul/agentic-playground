/**
 * "Who's chatting?" — pick a profile and type its 4-digit PIN, or make a new profile. A profile
 * made for someone without a PIN asks them to choose one the first time they pick it.
 * signIn() resolves with the signed-in profile; at once when this device already is.
 */
import { api } from './api.js';

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function initials(name) {
  return String(name || '?').trim().charAt(0).toUpperCase() || '?';
}

function show(view) {
  for (const id of ['signin-pick', 'pin-form', 'new-form']) $(id).hidden = id !== view;
  for (const id of ['pin-error', 'new-error']) $(id).hidden = true;
  const focus = { 'pin-form': 'pin-input', 'new-form': 'new-name' }[view];
  if (focus) setTimeout(() => $(focus).focus(), 0);
}

function fail(id, message) {
  $(id).textContent = message;
  $(id).hidden = false;
}

const digits = (input) => { input.value = input.value.replace(/\D/g, '').slice(0, 4); };

export async function signIn() {
  const { profiles, me, name } = await api('/profiles');
  if (me) return me;
  document.querySelectorAll('.kip-name').forEach(node => { node.textContent = name || 'Kip'; });
  $('signin').hidden = false;
  return new Promise((resolve) => {
    let chosen = null;
    let claiming = null; // a profile made without a PIN: its person chooses one now
    const newProfile = (profile = null) => {
      claiming = profile;
      $('new-name').hidden = $('new-name').labels[0].hidden = Boolean(profile);
      $('new-name').required = !profile;
      $('new-form').querySelector('h1').textContent = profile ? `Hi, ${profile.name}` : 'New profile';
      $('new-pin').labels[0].textContent = profile ? 'Choose your 4-digit PIN' : 'Choose a 4-digit PIN';
      $('new-form').querySelector('[type="submit"]').textContent = profile ? 'Set PIN' : 'Create';
      $('new-owner-note').hidden = Boolean(profile) || profiles.length > 0;
      show('new-form');
    };
    const done = (profile) => { $('signin').hidden = true; resolve(profile); };

    const list = $('profile-list');
    list.replaceChildren(...profiles.map((profile) => {
      const button = el('button', 'profile-button');
      button.type = 'button';
      button.append(el('span', 'me-avatar', initials(profile.name)), el('span', 'profile-name', profile.name));
      button.addEventListener('click', () => {
        if (profile.needs_pin) return newProfile(profile);
        chosen = profile;
        $('pin-name').textContent = `Hi, ${profile.name}`;
        $('pin-input').value = '';
        show('pin-form');
      });
      return button;
    }));
    $('profile-new').addEventListener('click', () => newProfile());
    document.querySelectorAll('#signin [data-back]').forEach(button => button.addEventListener('click', () => show('signin-pick')));
    if (profiles.length) show('signin-pick'); else newProfile();
    if (!profiles.length) document.querySelector('#new-form [data-back]').hidden = true;

    const pin = $('pin-input');
    const submitPin = async () => {
      if (pin.value.length !== 4 || pin.disabled) return;
      pin.disabled = true;
      try { done(await api('/signin', { profile_id: chosen.id, pin: pin.value })); }
      catch (error) { fail('pin-error', error.message); pin.value = ''; }
      finally { pin.disabled = false; if (!$('signin').hidden) pin.focus(); }
    };
    pin.addEventListener('input', () => { digits(pin); if (pin.value.length === 4) submitPin(); });
    $('pin-form').addEventListener('submit', (event) => { event.preventDefault(); submitPin(); });

    for (const id of ['new-pin', 'new-pin-again']) $(id).addEventListener('input', () => digits($(id)));
    $('new-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const nameValue = $('new-name').value.trim();
      const first = $('new-pin').value;
      if (!/^\d{4}$/.test(first)) return fail('new-error', 'The PIN is 4 digits.');
      if (first !== $('new-pin-again').value) return fail('new-error', 'The two PINs are different.');
      const submit = event.submitter || $('new-form').querySelector('[type="submit"]');
      submit.disabled = true;
      try {
        done(claiming
          ? await api(`/profiles/${encodeURIComponent(claiming.id)}/pin`, { pin: first })
          : await api('/profiles', { name: nameValue, pin: first }));
      }
      catch (error) { fail('new-error', error.message); }
      finally { submit.disabled = false; }
    });
  });
}
