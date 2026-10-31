export function PDFViewer({ src, pageNumber }: { src: string; pageNumber?: number }) {
  const hash = pageNumber ? `#page=${pageNumber}` : "";
  return (
    <div className="bracket-frame border border-hairline bg-panel p-1">
      <iframe src={`${src}${hash}`} title="PDF preview" className="h-[600px] w-full" data-testid="pdf-viewer" />
    </div>
  );
}
