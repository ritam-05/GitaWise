export function ChariotLineArt() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none mx-auto aspect-[435/573] w-28 select-none bg-[var(--line-art)] opacity-75 transition-colors duration-300 dark:opacity-80 sm:w-[8.5rem] lg:w-36"
      style={{
        maskImage: "url('/transparent_gita.png')",
        WebkitMaskImage: "url('/transparent_gita.png')",
        maskRepeat: "no-repeat",
        WebkitMaskRepeat: "no-repeat",
        maskPosition: "center",
        WebkitMaskPosition: "center",
        maskSize: "contain",
        WebkitMaskSize: "contain",
      }}
    />
  );
}
