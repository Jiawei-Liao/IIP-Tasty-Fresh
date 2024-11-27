import { useState, useEffect } from 'react'

export function useImageDimensions(imageRef, containerRef) {
    const [dimensions, setDimensions] = useState({
        width: 0,
        height: 0,
        offsetX: 0,
        offsetY: 0,
        imageAspectRatio: 1
    })

    useEffect(() => {
        const updateDimensions = () => {
            const img = imageRef.current
            const container = containerRef?.current || img?.parentElement
            if (!img || !container) return

            const containerRect = container.getBoundingClientRect()
            const imageAspectRatio = img.naturalWidth / img.naturalHeight
            const containerAspectRatio = containerRect.width / containerRect.height

            let width, height
            if (imageAspectRatio > containerAspectRatio) {
                width = containerRect.width
                height = containerRect.width / imageAspectRatio
            } else {
                height = containerRect.height
                width = containerRect.height * imageAspectRatio
            }

            const offsetX = (containerRect.width - width) / 2
            const offsetY = (containerRect.height - height) / 2

            setDimensions({
                width,
                height,
                offsetX,
                offsetY,
                imageAspectRatio
            })
        }

        const img = imageRef.current
        if (img?.complete) {
            updateDimensions()
        } else {
            img?.addEventListener('load', updateDimensions)
        }

        window.addEventListener('resize', updateDimensions)

        return () => {
            window.removeEventListener('resize', updateDimensions)
            img?.removeEventListener('load', updateDimensions)
        }
    }, [imageRef, containerRef])

    return dimensions
}