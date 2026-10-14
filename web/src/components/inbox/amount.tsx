/** A formatted amount ("AED 1,234.00") with its currency code set back in the muted tone, so the
 * figures carry a money column. The text content is unchanged, so copy and tests see one string. */
export function Amount({ text }: { text: string }) {
  const space = text.indexOf(" ");
  if (space !== 3 || !/^[A-Z]{3}$/.test(text.slice(0, 3))) return <>{text}</>;
  return (
    <>
      <span className="text-muted-foreground">{text.slice(0, 3)}</span> {text.slice(4)}
    </>
  );
}
