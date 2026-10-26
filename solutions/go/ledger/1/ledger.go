package ledger

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

type Entry struct {
	Date        string
	Description string
	Change      int
}

func FormatLedger(currency string, locale string, entries []Entry) (string, error) {
	var currSymbol string
	switch currency {
	case "USD":
		currSymbol = "$"
	case "EUR":
		currSymbol = "€"
	default:
		return "", errors.New("invalid currency")
	}

	var header, dateFormat string
	var thousandsSep, decimalSep byte
	switch locale {
	case "en-US":
		header = fmt.Sprintf("%-10s | %-25s | %-13s\n", "Date", "Description", "Change")
		dateFormat = "01/02/2006"
		thousandsSep, decimalSep = ',', '.'
	case "nl-NL":
		header = fmt.Sprintf("%-10s | %-25s | %-13s\n", "Datum", "Omschrijving", "Verandering")
		dateFormat = "02-01-2006"
		thousandsSep, decimalSep = '.', ','
	default:
		return "", errors.New("invalid locale")
	}

	sortedEntries := make([]Entry, len(entries))
	copy(sortedEntries, entries)

	slices.SortFunc(sortedEntries, func(a, b Entry) int {
		if c := strings.Compare(a.Date, b.Date); c != 0 {
			return c
		}
		if c := strings.Compare(a.Description, b.Description); c != 0 {
			return c
		}
		return a.Change - b.Change
	})

	var sb strings.Builder
	sb.WriteString(header)

	for _, entry := range sortedEntries {
		t, err := time.Parse("2006-01-02", entry.Date)
		if err != nil {
			return "", err
		}
		dateStr := t.Format(dateFormat)

		descRunes := []rune(entry.Description)
		var descStr string
		if len(descRunes) > 25 {
			descStr = string(descRunes[:22]) + "..."
		} else {
			descStr = fmt.Sprintf("%-25s", entry.Description)
		}

		changeStr := formatAmount(entry.Change, currSymbol, locale, thousandsSep, decimalSep)
		sb.WriteString(fmt.Sprintf("%-10s | %s | %13s\n", dateStr, descStr, changeStr))
	}

	return sb.String(), nil
}

func formatAmount(change int, symbol, locale string, thousandsSep, decimalSep byte) string {
	isNegative := change < 0
	absVal := change
	if isNegative {
		absVal = -absVal
	}

	cents := absVal % 100
	dollars := absVal / 100

	dollarStr := fmt.Sprintf("%d", dollars)
	var intParts []string
	for len(dollarStr) > 3 {
		intParts = append([]string{dollarStr[len(dollarStr)-3:]}, intParts...)
		dollarStr = dollarStr[:len(dollarStr)-3]
	}
	intParts = append([]string{dollarStr}, intParts...)
	formattedInt := strings.Join(intParts, string(thousandsSep))

	var res string
	switch locale {
	case "en-US":
		if isNegative {
			res = fmt.Sprintf("(%s%s%c%02d)", symbol, formattedInt, decimalSep, cents)
		} else {
			res = fmt.Sprintf("%s%s%c%02d ", symbol, formattedInt, decimalSep, cents)
		}
	case "nl-NL":
		if isNegative {
			res = fmt.Sprintf("%s -%s%c%02d ", symbol, formattedInt, decimalSep, cents)
		} else {
			res = fmt.Sprintf("%s %s%c%02d ", symbol, formattedInt, decimalSep, cents)
		}
	}

	return res
}