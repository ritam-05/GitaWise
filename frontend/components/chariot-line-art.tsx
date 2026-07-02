import transparentGita from "@/public/transparent_gita.png";

export function ChariotLineArt() {
  return (
    <img
      src={transparentGita.src}
      alt=""
      aria-hidden="true"
      width={transparentGita.width}
      height={transparentGita.height}
      className="hero-line-art pointer-events-none mx-auto h-auto w-28 select-none object-contain opacity-80 transition-opacity duration-300 dark:opacity-85 sm:w-[8.5rem] lg:w-36"
    />
  );
}
