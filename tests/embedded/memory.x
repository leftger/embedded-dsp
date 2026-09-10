/* Simple run-from-ram layout for cycle counting */
MEMORY
{
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K  /* DTCM */
}
REGION_ALIAS("FLASH", RAM);
