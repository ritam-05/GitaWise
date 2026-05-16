import Image from "next/image";

export function HeroLineArt() {
  return (
    <Image
      src="/transparent_gita.png"
      alt=""
      width={553}
      height={702}
      priority
      aria-hidden="true"
      className="hero-line-art pointer-events-none mx-auto h-auto w-24 select-none object-contain opacity-70 transition-opacity duration-300 dark:opacity-60 sm:w-28 lg:w-32"
    />
  );
}
