create table if not exists public.cache_entries (
  key text primary key,
  value jsonb not null,
  expires_at timestamptz null,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists cache_entries_expires_at_idx
  on public.cache_entries (expires_at);

create or replace function public.set_cache_entries_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

drop trigger if exists cache_entries_set_updated_at on public.cache_entries;

create trigger cache_entries_set_updated_at
before update on public.cache_entries
for each row
execute function public.set_cache_entries_updated_at();
